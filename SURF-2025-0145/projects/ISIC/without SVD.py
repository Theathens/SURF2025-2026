import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import cv2
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import time
import argparse
import json
import warnings


warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Detect and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # GPU memory protection - limit maximum memory usage
    torch.cuda.set_per_process_memory_fraction(0.8)  # Only use 80% of available memory

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# Resource monitoring
def log_resources():
    if device.type == "cuda":
        alloc_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU memory usage: {alloc_mem:.2f}/{total_mem:.2f} GB ({alloc_mem / total_mem:.1%})")

    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        print(f"CPU usage: {cpu_percent}% | RAM usage: {ram_percent}%")
    except ImportError:
        pass


# Hyperparameter configuration
print("Enter number of training epochs:")
num = int(input())


class Config:
    IMAGE_SIZE = (384, 384)  # Adjusted to 384x384 for balance between accuracy and memory
    BATCH_SIZE = 16  # Batch size
    ACCUMULATION_STEPS = 2  # Gradient accumulation steps, equivalent to BATCH_SIZE=32
    EPOCHS = num
    LEARNING_RATE = 3e-5  # Lower learning rate for more complex ISIC data
    WEIGHT_DECAY = 1e-5
    BASE_MODEL = 'efficientnet_b4'
    PATIENCE = 15  # More patience for ISIC training
    MIN_LR = 1e-7
    DATA_DIR = r"D:\desktop\ISIC_2019"  # ISIC2019 dataset path, adjust manually
    TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "ISIC_2019_Training_Input")
    TEST_IMAGE_DIR = os.path.join(DATA_DIR, "ISIC_2019_Test_Input")
    OUTPUT_DIR = r"C:\Users\27785\PycharmProjects\PythonProject1\.venv\ISIC"  # Output path for results, adjust manually
    USE_MIXED_PRECISION = True


config = Config()


def load_and_analyze_data(data_dir):
    """Load and analyze ISIC2019 dataset"""
    print("Loading data...")
    # Typical CSV file names for ISIC2019
    train_path = os.path.join(data_dir, "ISIC_2019_Training_GroundTruth.csv")
    test_path = os.path.join(data_dir, "ISIC_2019_Test_GroundTruth.csv")

    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training CSV file not found: {train_path}")

    # Load training data
    train = pd.read_csv(train_path)
    print(f"Data loaded, training set size: {len(train)}")

    # ISIC2019 labels are in separate columns, need to convert to single label column
    # Define the 7 ISIC2019 classes
    classes = [
        'MEL',  # Melanoma
        'NV',  # Nevus
        'BCC',  # Basal cell carcinoma
        'AK',  # Actinic keratosis
        'BKL',  # Benign keratosis
        'DF',  # Dermatofibroma
        'VASC'  # Vascular lesion
    ]

    # Create label column by taking the column name with maximum value
    train['label'] = train[classes].idxmax(axis=1)

    # Process test set (ISIC test sets typically don't have public labels)
    if os.path.exists(test_path):
        test = pd.read_csv(test_path)
        print(f"Test set size: {len(test)}")
    else:
        # If no test metadata, create from test image directory
        test_images_dir = os.path.join(data_dir, "ISIC_2019_Test_Input")
        if os.path.exists(test_images_dir):
            test_filenames = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            test = pd.DataFrame({'image': test_filenames})
            print(f"Created test set from directory, {len(test)} images total")
        else:
            raise FileNotFoundError(f"Test set not found: {test_path} or {test_images_dir}")

    # Data integrity verification
    required_train_columns = {'image'} | set(classes)
    assert required_train_columns.issubset(train.columns), "Training data missing required columns"
    assert 'image' in test.columns, "Test data missing 'image' column"

    # Class distribution analysis
    class_counts = train['label'].value_counts()
    print(f"Class distribution:\n{class_counts}")

    # Calculate class weights (ISIC data is significantly imbalanced)
    unique_classes = np.unique(train['label'])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train['label']
    )

    # Apply stronger weights to minority classes (ISIC has significant imbalance)
    max_weight = max(class_weights)
    for i, weight in enumerate(class_weights):
        if weight < max_weight * 0.1:  # Enhance weights for very underrepresented classes
            class_weights[i] *= 5.0
        elif weight < max_weight * 0.3:
            class_weights[i] *= 3.0
        elif weight < max_weight * 0.5:
            class_weights[i] *= 2.0

    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Calculated class weights:", class_weights)

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    return train, test, class_weights, unique_classes


def enhanced_preprocessing(image_path):
    """Enhanced preprocessing pipeline for skin lesion images"""
    try:
        # 使用更快的图像读取方式
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return np.zeros((*config.IMAGE_SIZE, 3), dtype=np.float32) * 0.5

        # Convert to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        # Specialized contrast enhancement for skin lesions
        # Use CLAHE for contrast enhancement (particularly effective for skin lesions)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab_clahe = cv2.merge((cl, a, b))
        image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        # Reduce noise while preserving edges
        image = cv2.bilateralFilter(image, 9, 75, 75)

        # Resize image
        image = cv2.resize(image, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # Normalize
        return image.astype('float32') / 255.0

    except Exception as e:
        print(f"Preprocessing error: {e}")
        # Error recovery - use gray image as replacement
        image = cv2.imread(image_path)
        if image is None:
            image = np.ones((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), dtype=np.float32) * 0.5
        else:
            image = cv2.resize(image, config.IMAGE_SIZE)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype('float32') / 255.0


class ISICDataset(Dataset):
    """ISIC dataset class"""

    def __init__(self, dataframe, directory, x_col='image', y_col=None, transform=None):
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.transform = transform

        if y_col:
            self.classes = sorted(dataframe[y_col].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            print(f"Dataset class mapping: {self.class_to_idx}")
        else:
            self.classes = []
            self.class_to_idx = {}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # ISIC images are typically stored as .jpg or .png
        image_filename = row[self.x_col]
        if not image_filename.endswith(('.jpg', '.png', '.jpeg')):
            image_filename += '.jpg'  # Handle cases where extension might be missing

        image_path = os.path.join(self.directory, image_filename)
        image = enhanced_preprocessing(image_path)

        if self.transform:
            image = self.transform(image)

        if self.y_col:
            label = self.class_to_idx[row[self.y_col]]
            return image, label
        else:
            return image, row[self.x_col]


def create_data_loaders(train_df, test_df, class_weights, data_dir):
    """Create data loaders"""
    print("Creating data loaders...")

    # Training data augmentation (adjusted for skin lesion images)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),  # Larger rotation angle
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stronger color jitter
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation/test data transformation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Split training set into training and validation sets - ensure stratified sampling
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['label'] if 'label' in train_df.columns else None
    )
    print(f"Dataset split complete, training set: {len(train_df)}, validation set: {len(val_df)}")

    # Create datasets
    # ISIC2019 images are typically stored in train and test subdirectories
    train_dataset = ISICDataset(
        train_df,
        config.TRAIN_IMAGE_DIR,  # 验证集也使用训练图像路径
        'image',
        'label',
        transform=train_transform
    )

    val_dataset = ISICDataset(
        val_df,
        config.TRAIN_IMAGE_DIR,  # 验证集也使用训练图像路径
        'image',
        'label',
        transform=val_transform
    )

    test_dataset = ISICDataset(
        test_df,
        config.TEST_IMAGE_DIR,  # 使用正确的测试图像路径
        'image',
        None,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2 if device.type != 'cuda' else 0,  # Use 0 workers for CUDA devices
        pin_memory=True,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2 if device.type != 'cuda' else 0,
        pin_memory=True,
        persistent_workers=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2 if device.type != 'cuda' else 0,
        pin_memory=True,
        persistent_workers=False
    )

    print("Data loaders created")
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def build_advanced_model(num_classes):
    """Build enhanced EfficientNet-B4 model (adjusted for multi-class classification)"""
    print("Building model...")
    from torchvision.models import EfficientNet_B4_Weights

    # Load pre-trained EfficientNet-B4 model
    base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    features = base_model.features
    print("Pre-trained model loaded")

    # Unfreeze more layers to adapt to complex ISIC features
    for param in features.parameters():
        param.requires_grad = False
    # For ISIC data, unfreeze more layers to capture complex skin lesion features
    for param in features[-15:].parameters():  # Unfreeze 5 more layers than original
        param.requires_grad = True
    print("Model parameters unfrozen")

    # Build feature extractor
    feature_extractor = nn.Sequential(
        features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    ).to(device)

    # Calculate feature dimension
    with torch.no_grad():
        sample_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(device)
        features_dim = feature_extractor(sample_input).shape[1]
    print(f"Actual feature dimension: {features_dim}")

    # Build classifier head (for 7-class classification task)
    classifier = nn.Sequential(
        nn.Linear(features_dim, 1024),
        nn.SiLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),

        nn.Linear(1024, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),

        nn.Linear(512, 256),
        nn.SiLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.5),

        nn.Linear(256, num_classes),
    ).to(device)

    # Combine full model
    full_model = nn.Sequential(
        feature_extractor,
        classifier
    ).to(device)

    # Print model statistics
    total_params = sum(p.numel() for p in full_model.parameters())
    trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}, trainable parameters: {trainable_params:,}")

    return full_model


def train_model(model, train_loader, val_loader, class_weights, unique_classes):
    """Train the model (adjusted for multi-class task)"""

    print("Enter learning rate patience:")
    n = int(input())

    print("Starting model training...")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Learning rate schedulers
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=n,
                                                      min_lr=config.MIN_LR)

    # Early stopping
    best_val_auc = 0.0
    patience = config.PATIENCE
    counter = 0
    best_epoch = 0

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_auc': [],
        'val_auc': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': []
    }

    # Record start time
    start_time = time.time()
    epoch_times = []

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch + 1}/{config.EPOCHS} =====")

        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_all_preds = []
        train_all_labels = []
        train_all_probs = []

        print("Training...")
        for i, (inputs, labels) in enumerate(train_loader):
            # Manage memory usage
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=config.USE_MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / config.ACCUMULATION_STEPS  # Gradient accumulation

            loss.backward()

            # Update parameters every accumulation_steps
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * config.ACCUMULATION_STEPS * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0) * config.ACCUMULATION_STEPS
            correct += predicted.eq(labels).sum().item() * config.ACCUMULATION_STEPS

            # Collect training predictions for metrics calculation
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())

            # Collect probabilities for AUC calculation
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            train_all_probs.extend(probs)

            # Print progress
            if (i + 1) % 20 == 0:
                print(
                    f"Processing batch {i + 1}/{len(train_loader)}, current batch loss: {loss.item() * config.ACCUMULATION_STEPS:.4f}")

            # Periodically clear cache
            if (i + 1) % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        # Calculate training precision and recall
        train_micro_precision = precision_score(train_all_labels, train_all_preds, average='micro')
        train_micro_recall = recall_score(train_all_labels, train_all_preds, average='micro')

        # Calculate training AUC (multi-class)
        num_classes = len(unique_classes)
        train_labels_onehot = label_binarize(train_all_labels, classes=list(range(num_classes)))
        train_auc = roc_auc_score(train_labels_onehot, train_all_probs, average='macro', multi_class='ovr')

        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_accuracy = 100. * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['train_precision'].append(train_micro_precision)
        history['train_recall'].append(train_micro_recall)
        history['train_auc'].append(train_auc)

        print(f"Training loss: {epoch_loss:.4f}, training accuracy: {epoch_accuracy:.2f}%")
        print(f"Training precision: {train_micro_precision:.4f}, training recall: {train_micro_recall:.4f}")
        print(f"Training AUC: {train_auc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        print("Validating...")
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Manage memory usage
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        # Calculate validation precision and recall
        val_micro_precision = precision_score(all_labels, all_preds, average='micro')
        val_micro_recall = recall_score(all_labels, all_preds, average='micro')

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = 100. * correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['val_precision'].append(val_micro_precision)
        history['val_recall'].append(val_micro_recall)
        print(f"Validation loss: {epoch_val_loss:.4f}, validation accuracy: {epoch_val_accuracy:.2f}%")
        print(f"Validation precision: {val_micro_precision:.4f}, validation recall: {val_micro_recall:.4f}")

        # Calculate AUC (multi-class)
        num_classes = len(unique_classes)
        all_labels_onehot = label_binarize(all_labels, classes=list(range(num_classes)))
        val_auc = roc_auc_score(all_labels_onehot, all_probs, average='macro', multi_class='ovr')
        history['val_auc'].append(val_auc)
        print(f"Validation macro-average AUC: {val_auc:.4f}")

        # Update learning rate
        scheduler1.step()
        scheduler2.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = (config.EPOCHS - epoch - 1) * avg_epoch_time
        print(f"Epoch time: {epoch_time:.2f}s, average: {avg_epoch_time:.2f}s, remaining: {remaining_time / 60:.1f}min")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            counter = 0
            model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved to {model_path}, validation AUC: {best_val_auc:.4f}")
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered, best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
                break

        # Clear cache after each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    total_time = time.time() - start_time
    print(f"Training completed, total time: {(time.time() - start_time) / 60:.2f} minutes")

    # Save training time information
    time_info = {
        'total_time': total_time,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'epoch_times': epoch_times
    }
    with open(os.path.join(config.OUTPUT_DIR, 'training_time.json'), 'w') as f:
        json.dump(time_info, f)

    return history


def evaluate_model(model, test_loader, class_to_idx, output_dir, unique_classes, device):
    """Evaluate model and generate detailed report"""
    print("Starting model evaluation...")
    model.eval()
    results = []
    all_preds = []
    all_filenames = []
    all_probs = []

    with torch.no_grad():
        for inputs, filenames in test_loader:
            # Manage memory usage
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            inputs = inputs.to(device)

            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            idx_to_class = {v: k for k, v in class_to_idx.items()}
            for i, filename in enumerate(filenames):
                pred_class = idx_to_class[preds[i]]
                # Add probabilities for each class for ISIC results
                class_prob = {cls: float(probs[i][idx]) for cls, idx in class_to_idx.items()}
                results.append({
                    'filename': filename,
                    'prediction': pred_class, **class_prob
                })

            all_preds.extend(preds)
            all_filenames.extend(filenames)
            all_probs.extend(probs)

            # Periodically clear cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Save results (ISIC format)
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'isic_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation completed, results saved to {results_path}")

    # Generate class distribution histogram
    plt.figure(figsize=(10, 6))
    pred_classes = [idx_to_class[p] for p in all_preds]
    sns.countplot(x=pred_classes)
    plt.title('Predicted Class Distribution on Test Set')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    plt.close()

    return results


def visualize_results(history, output_dir, class_names=None):
    """Enhanced visualization of training metrics"""
    plt.figure(figsize=(18, 12))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Accuracy curve
    plt.subplot(2, 2, 1)
    if 'train_accuracy' in history and 'val_accuracy' in history:
        plt.plot(history['train_accuracy'], label='Training Accuracy', marker='o')
        plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
        plt.title('Accuracy Over Epochs')
        plt.ylabel('Accuracy (%)')
    else:
        plt.text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Accuracy Data Missing')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Loss curve
    plt.subplot(2, 2, 2)
    if 'train_loss' in history and 'val_loss' in history:
        plt.plot(history['train_loss'], label='Training Loss', marker='o')
        plt.plot(history['val_loss'], label='Validation Loss', marker='s')
        plt.title('Loss Over Epochs')
        plt.ylabel('Loss')
    else:
        plt.text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Loss Data Missing')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. AUC curve
    plt.subplot(2, 2, 3)
    if 'train_auc' in history and 'val_auc' in history:
        plt.plot(history['train_auc'], label='Training AUC', marker='o')
        plt.plot(history['val_auc'], label='Validation AUC', marker='s')
        plt.title('AUC Over Epochs')
        plt.ylabel('AUC')
    elif 'train_macro_auc' in history and 'val_macro_auc' in history:
        plt.plot(history['train_macro_auc'], label='Training Macro-Average AUC', marker='o')
        plt.plot(history['val_macro_auc'], label='Validation Macro-Average AUC', marker='s')
        plt.title('Macro-Average AUC Over Epochs')
        plt.ylabel('AUC')
    else:
        plt.text(0.5, 0.5, 'AUC data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('AUC Data Missing')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. Precision and recall curve
    plt.subplot(2, 2, 4)
    if 'train_precision' in history and 'val_precision' in history and 'train_recall' in history and 'val_recall' in history:
        plt.plot(history['train_precision'], label='Training Precision', marker='o')
        plt.plot(history['val_precision'], label='Validation Precision', marker='s')
        plt.plot(history['train_recall'], label='Training Recall', marker='^')
        plt.plot(history['val_recall'], label='Validation Recall', marker='v')
        plt.title('Precision and Recall Over Epochs')
        plt.ylabel('Metric Value')
    else:
        plt.text(0.5, 0.5, 'Precision/Recall data not available', ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.title('Precision/Recall Data Missing')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save figure
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training metrics visualization saved to: {plot_path}")


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='ISIC2019 Skin Lesion Image Classification Model')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help='Output directory')
    args = parser.parse_args()

    try:
        # Update configuration
        config.DATA_DIR = args.data_dir
        config.OUTPUT_DIR = args.output_dir

        # Load data
        train_df, test_df, class_weights, unique_classes = load_and_analyze_data(config.DATA_DIR)

        # Create data loaders
        train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
            train_df, test_df, class_weights, config.DATA_DIR
        )

        # Build model
        model = build_advanced_model(len(class_to_idx))

        # Train model
        history = train_model(model, train_loader, val_loader, class_weights, unique_classes)

        # Visualize results
        visualize_results(history, config.OUTPUT_DIR, class_names=class_to_idx.keys())

        # Load best model for evaluation
        best_model = build_advanced_model(len(class_to_idx))
        best_model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth')))
        best_model.to(device).eval()

        # Evaluate model
        evaluate_model(best_model, test_loader, class_to_idx, config.OUTPUT_DIR, unique_classes, device)

        # Save final model
        final_model_path = os.path.join(config.OUTPUT_DIR, 'final_isic_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Model saved as {final_model_path}")

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
