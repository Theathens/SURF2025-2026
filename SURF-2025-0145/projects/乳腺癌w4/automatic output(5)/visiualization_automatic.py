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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import time
import argparse
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Detect and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    # GPU memory protection - limit maximum usage
    torch.cuda.set_per_process_memory_fraction(0.8)  # only 80% of the memory could be used
    # Configure memory allocator to avoid fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


class Config:
    IMAGE_SIZE = (384, 384)  # Adjusted to 384 x 384 for balance between accuracy and memory
    BATCH_SIZE = 16  # Batch size
    ACCUMULATION_STEPS = 2  # Gradient accumulation steps, equivalent to BATCH_SIZE=32
    LEARNING_RATE = 5e-5  # Reduced learning rate
    WEIGHT_DECAY = 1e-5
    BASE_MODEL = 'efficientnet_b4'
    MIN_LR = 1e-7
    DATA_DIR = r"D:\desktop\w3\w3\mammography_images"  # Input directory for training and test images
    OUTPUT_DIR = r"C:\Users\27785\PycharmProjects\PythonProject1\.venv\乳腺癌w4\automatic output(5)"  # Output directory for results
    USE_MIXED_PRECISION = True
    USE_SVD = True  # Enable SURF-2025-0145 denoising
    # Threshold for singular value change rate
    SVD_CHANGE_THRESHOLD = 0.1  # Default threshold for adaptive SURF-2025-0145

    EPOCHS = None
    PATIENCE = None


config = Config()


# Adaptive SURF-2025-0145 denoising based on singular value change rate
def svd_change_rate_adjustment(image, change_threshold=None):
    """
    Apply adaptive SURF-2025-0145 denoising on each RGB channel separately
    to preserve color information critical for medical imaging.
    Returns:
        denoised_image: denoised image
        k_avg: average k value across channels
    """
    if change_threshold is None:
        change_threshold = config.SVD_CHANGE_THRESHOLD

    # Handle uniform images
    if np.max(image) == np.min(image):
        return image.astype(np.float32), 0  # Return k=0 for uniform images

    # Normalize across all channels
    min_val, max_val = np.min(image), np.max(image)
    normalized = (image - min_val) / (max_val - min_val + 1e-8)

    denoised_channels = []
    k_values = []  # Store k values for each channel
    for c in range(3):  # Process each RGB channel independently
        channel = normalized[..., c].astype(np.float32)
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)

        # Calculate adaptive k based on singular value ratios
        if len(S) > 1:
            ratios = S[:-1] / S[1:]
            valid_indices = np.where(ratios < (1 + change_threshold))[0]
            k = valid_indices[0] + 1 if len(valid_indices) > 0 else max(1, int(len(S) * 0.2))
        else:
            k = 1

        # Ensure k retains at least 10% of singular values (critical for medical details)
        k = max(int(len(S) * 0.1), k)  # Minimum 10% retention
        k = min(k, len(S))  # Ensure k is within valid range

        k_values.append(k)  # Record k value for this channel

        # Reconstruct channel with truncated SURF-2025-0145
        S_trunc = np.diag(S[:k])
        reconstructed_channel = U[:, :k] @ S_trunc @ Vt[:k, :]
        denoised_channels.append(reconstructed_channel)

    # Calculate average k value across channels
    k_avg = sum(k_values) / len(k_values)

    # Combine channels and denormalize
    denoised = np.stack(denoised_channels, axis=-1)
    denoised = denoised * (max_val - min_val) + min_val
    return np.clip(denoised, 0, 255).astype(np.float32), k_avg


def enhanced_preprocessing(image_path):
    """
    Enhanced image preprocessing pipeline optimized for medical images.
    Preserves color channels during SURF-2025-0145 processing to retain diagnostic details.
    Returns:
        preprocessed_image: normalized image
        k_value: k value used in SURF-2025-0145 (0 if SURF-2025-0145 not used)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Enhance contrast in LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        k_value = 0
        # Apply SURF-2025-0145 denoising on RGB channels
        if config.USE_SVD:
            image, k_value = svd_change_rate_adjustment(image.astype(np.float32))

        # Gaussian blur, resize, and normalize
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.resize(image, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return image.astype('float32') / 255.0, k_value

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
        return image.astype('float32') / 255.0, 0  # Return k=0 for error cases


def visualize_svd_effect(image_path, output_dir):
    """Visualize SURF-2025-0145 denoising effect for sample images"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Apply SURF-2025-0145 denoising (on RGB channels)
        denoised, k_value = svd_change_rate_adjustment(blurred.astype(np.float32))
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)

        # Resize for display
        display_size = (384, 384)
        orig_display = cv2.resize(image, display_size)
        blur_display = cv2.resize(blurred, display_size)
        svd_display = cv2.resize(denoised, display_size)

        # Create comparison plot
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(orig_display)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(blur_display)
        plt.title('Blurred Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(svd_display)
        plt.title(f'SURF-2025-0145 Denoised (k={k_value:.1f})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'svd_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SURF-2025-0145 comparison visualization saved to {os.path.join(output_dir, 'svd_comparison.png')}")

    except Exception as e:
        print(f"Error in SURF-2025-0145 visualization: {e}")


class MammographyDataset(Dataset):
    """Mammography image dataset class"""

    def __init__(self, dataframe, directory, x_col, y_col=None, transform=None):
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
        image_path = os.path.join(self.directory, row[self.x_col])
        image, k_value = enhanced_preprocessing(image_path)

        if self.transform:
            image = self.transform(image)

        if self.y_col:
            label = self.class_to_idx[row[self.y_col]]
            return image, label, k_value
        else:
            return image, row[self.x_col], k_value


def load_and_analyze_data(data_dir):
    """Load and analyze dataset"""
    print("Loading data...")
    train_path = r"D:\desktop\w2\w2\mammography_images\Training_set.csv"  # Adjust manually
    test_path = r"D:\desktop\w2\w2\mammography_images\Testing_set.csv"  # Adjust manually

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Training or testing CSV file not found: {train_path}, {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Data loaded, training set size: {len(train)}, test set size: {len(test)}")

    # Data integrity verification
    required_columns = {'filename', 'label'}
    assert required_columns.issubset(train.columns), "Training data is missing required columns"
    assert 'filename' in test.columns, "Test data is missing filename column"

    # Class distribution analysis
    class_counts = train['label'].value_counts()
    print(f"Class distribution:\n{class_counts}")

    # Calculate class weights
    unique_classes = np.unique(train['label'])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train['label']
    )

    # Apply stronger weights to minority classes
    max_weight = max(class_weights)
    for i, weight in enumerate(class_weights):
        if weight < max_weight * 0.2:  # Enhance weights for particularly underrepresented classes
            class_weights[i] *= 3.0
        elif weight < max_weight * 0.5:
            class_weights[i] *= 2.0

    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Class weights calculated:", class_weights)

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Visualize SURF-2025-0145 effect on sample images
    sample_image = os.path.join(data_dir, 'train', train.iloc[0]['filename'])
    visualize_svd_effect(sample_image, config.OUTPUT_DIR)

    return train, test, class_weights, unique_classes


def create_data_loaders(train_df, test_df, class_weights, data_dir):
    """Create data loaders with optimized worker configuration"""
    print("Creating data loaders...")

    # Training data augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation/test data augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Split training set into training and validation sets
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['label'] if 'label' in train_df.columns else None
    )
    print(f"Dataset split completed, training set: {len(train_df)}, validation set: {len(val_df)}")

    # Create datasets
    train_dataset = MammographyDataset(
        train_df,
        os.path.join(data_dir, 'train'),
        'filename',
        'label',
        transform=train_transform
    )

    val_dataset = MammographyDataset(
        val_df,
        os.path.join(data_dir, 'train'),
        'filename',
        'label',
        transform=val_transform
    )

    test_dataset = MammographyDataset(
        test_df,
        os.path.join(data_dir, 'test'),
        'filename',
        None,
        transform=val_transform
    )

    # Optimized data loaders with increased workers for better performance
    num_workers = 4 if device.type == 'cuda' else 2  # Increase workers for faster data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True  # Keep workers alive for faster iteration
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    print("Data loaders created")
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def build_advanced_model(num_classes):
    """Build enhanced EfficientNet-B4 model"""
    print("Building model...")
    from torchvision.models import EfficientNet_B4_Weights

    # Load pre-trained EfficientNet-B4 model
    base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    features = base_model.features
    print("Pre-trained model loaded")

    # Unfreeze more layers for fine-tuning
    for param in features.parameters():
        param.requires_grad = False
    for param in features[-10:].parameters():  # Unfreeze last 10 layers
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

    # Build classifier head with multiple layers and batch normalization
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
    """Train the model with early stopping and learning rate scheduling"""
    print("Starting model training...")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Learning rate schedulers
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config.PATIENCE,
                                                      min_lr=config.MIN_LR)

    # Early stopping configuration
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
        'val_recall': [],
        'train_k_values': []  # Store average k values per epoch
    }

    # Record start time
    start_time = time.time()
    epoch_times = []

    for epoch in range(config.EPOCHS):
        torch.cuda.empty_cache()
        gc.collect()

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
        epoch_k_values = []  # Store k values for this epoch

        print("Training...")
        for i, (inputs, labels, k_values) in enumerate(train_loader):
            # Collect k values
            epoch_k_values.extend(k_values.numpy())

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

            # Collect training predictions for AUC
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            train_all_probs.extend(probs)

            # Print progress
            if (i + 1) % 20 == 0:
                print(
                    f"Processing batch {i + 1}/{len(train_loader)}, current batch loss: {loss.item() * config.ACCUMULATION_STEPS:.4f}")

            # Clean cache periodically
            if (i + 1) % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

        # Calculate average k value for this epoch
        if epoch_k_values:
            avg_k = np.mean(epoch_k_values)
            history['train_k_values'].append(avg_k)
            print(f"Average k value for epoch {epoch + 1}: {avg_k:.2f}")
        else:
            history['train_k_values'].append(0)
            print("No k values recorded for this epoch")

        # Calculate training metrics
        train_micro_precision = precision_score(train_all_labels, train_all_preds, average='micro')
        train_micro_recall = recall_score(train_all_labels, train_all_preds, average='micro')

        # Calculate training AUC
        num_classes = len(unique_classes)
        if num_classes == 2:
            train_auc = roc_auc_score(train_all_labels, [p[1] for p in train_all_probs])
        else:
            from sklearn.preprocessing import label_binarize
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
            for inputs, labels, _ in val_loader:  # Ignore k values for validation
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

        # Calculate validation metrics
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

        # Calculate AUC
        num_classes = len(unique_classes)

        if num_classes == 2:
            val_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
            history['val_auc'].append(val_auc)
            print(f"Validation AUC: {val_auc:.4f}")
        else:
            from sklearn.preprocessing import label_binarize
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
        print(f"Epoch time: {epoch_time:.2f}s, avg: {avg_epoch_time:.2f}s, remaining: {remaining_time / 60:.1f}min")

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
            print(f"Early stopping counter: {counter}/{config.PATIENCE}")
            if counter >= config.PATIENCE:
                print(f"Early stopping triggered, best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
                break

        # Clean cache after each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    # Generate confusion matrix for validation results
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Validation Set)')
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix saved to {os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')}")

    total_time = time.time() - start_time
    print(f"Training completed, total time: {total_time / 60:.2f} minutes")

    # Save training time information
    time_info = {
        'total_time': total_time,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'epoch_times': epoch_times
    }
    with open(os.path.join(config.OUTPUT_DIR, 'training_time.json'), 'w') as f:
        json.dump(time_info, f)

    # Visualize k values over epochs
    visualize_k_values(history['train_k_values'], config.OUTPUT_DIR)

    return history


def visualize_k_values(k_values, output_dir):
    """Visualize k values over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(k_values) + 1), k_values, marker='o', linestyle='-', color='b')
    plt.title('Average k Value per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average k Value')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate points with their values
    for i, k in enumerate(k_values):
        plt.annotate(f'{k:.1f}', (i + 1, k), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'k_values_per_epoch.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"K values visualization saved to: {plot_path}")


def evaluate_model(model, test_loader, class_to_idx, output_dir, unique_classes, device):
    """Evaluate the model and generate detailed predictions"""
    print("Starting model evaluation...")
    model.eval()
    results = []
    all_preds = []
    all_labels = []  # Note: Test set may not have labels, storing filenames instead
    all_filenames = []
    all_probs = []

    with torch.no_grad():
        for inputs, filenames, k_values in test_loader:
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
                results.append({
                    'filename': filename,
                    'prediction': pred_class,
                    'probabilities': probs[i].tolist(),
                    'k_value': k_values[i].item()  # Record k value for each image
                })

            all_preds.extend(preds)
            all_filenames.extend(filenames)
            all_probs.extend(probs)

            # Clean cache periodically
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation completed, results saved to {results_path}")

    return results


def visualize_results(history, output_dir):
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


# Add Grad-CAM visualization for model interpretability (medical-specific)
def visualize_attention(model, test_loader, output_dir, class_to_idx):
    """
    Visualize model attention using Grad-CAM to highlight regions critical for prediction.
    This is particularly important for medical image analysis.
    """
    try:
        from gradcam import GradCAM
        from gradcam.utils import visualize_cam

        os.makedirs(os.path.join(output_dir, 'gradcam'), exist_ok=True)
        print("Generating Grad-CAM visualizations...")

        # Target the last convolutional layer for Grad-CAM
        target_layer = model[0][-1][-1][-1]  # Adjust based on model structure
        gradcam = GradCAM(model, target_layer)

        idx_to_class = {v: k for k, v in class_to_idx.items()}

        model.eval()
        for inputs, filenames, _ in test_loader:  # Ignore k values for Grad-CAM
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            for i in range(len(inputs)):
                input_tensor = inputs[i].unsqueeze(0)
                mask, _ = gradcam(input_tensor)
                heatmap, result = visualize_cam(mask, input_tensor)

                # Save visualization
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(inputs[i].cpu().permute(1, 2, 0).numpy())
                plt.title('Original Image')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(result.cpu().permute(1, 2, 0).numpy())
                plt.title(f'Prediction: {idx_to_class[preds[i].item()]}')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'gradcam', f'{filenames[i]}.png'))
                plt.close()

        print(f"Grad-CAM visualizations saved to {os.path.join(output_dir, 'gradcam')}")

    except ImportError:
        print("Grad-CAM library not installed. Please install it using: pip install grad-cam")
    except Exception as e:
        print(f"Error generating Grad-CAM visualizations: {e}")


def main():
    """Main function to orchestrate model training and evaluation"""
    # 打印设备信息（只执行一次）
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    # 获取用户输入
    print("How many epochs are needed:")
    config.EPOCHS = int(input())

    print("Please enter the epoch patience:")
    config.PATIENCE = int(input())

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Mammography Image Classification Model')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--use_svd', action='store_true', help='Enable SURF-2025-0145 denoising')
    parser.add_argument('--svd_components', type=int, default=50, help='Number of SURF-2025-0145 components to retain')
    args = parser.parse_args()

    try:
        # Update configuration
        config.DATA_DIR = args.data_dir
        config.OUTPUT_DIR = args.output_dir
        config.USE_SVD = args.use_svd
        config.SVD_COMPONENTS = args.svd_components

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
        visualize_results(history, config.OUTPUT_DIR)

        # Load best model for evaluation
        best_model = build_advanced_model(len(class_to_idx))
        best_model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth')))
        best_model.to(device).eval()

        # Evaluate model
        evaluate_model(best_model, test_loader, class_to_idx, config.OUTPUT_DIR, unique_classes, device)

        # Generate Grad-CAM visualizations for interpretability
        visualize_attention(best_model, test_loader, config.OUTPUT_DIR, class_to_idx)

        # Save final model
        final_model_path = os.path.join(config.OUTPUT_DIR, 'final_advanced_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Model saved as {final_model_path}")

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()