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

"""
Before running this programme, please check line 342, line 343, line 885 and line 887 and make sure they are adjusted manually based on your own device

line 347 should be where the table named Training_set.csv is saved locally
line 348 should be where the table named Testing_set.csv is saved locally
line 889 should be where the folder named mammography_images is saved locally (with both the train and test set of images included)
line 891 should be where you want to save the programme output
"""


warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Detect and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    # GPU memory protection - limit maximun use
    torch.cuda.set_per_process_memory_fraction(0.8)  # only 80% of the memory could be used
    # Configure memory allocator to avoid fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# vigilite the using
def log_resources():
    if device.type == "cuda":
        alloc_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU memory using: {alloc_mem:.2f}/{total_mem:.2f} GB ({alloc_mem / total_mem:.1%})")

    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        print(f"CPU using: {cpu_percent}% | memory using: {ram_percent}%")
    except ImportError:
        pass


# identify UNet construction
class UNetDenoiser(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()

        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # Residual connection enhancement
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Original input for residual connection
        identity = x

        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck layer
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output layer + residual connection
        return torch.sigmoid(self.conv(dec1) + self.residual(identity))


    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True)
        )


# Load the UNet model
def load_unet_model(weights_path):
    """Load the pre-trained UNet denoising model"""
    model = UNetDenoiser(in_channels=1, out_channels=1).to(device)

    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded UNet weights: {weights_path}")
            return model
        except Exception as e:
            print(f"Failed to load UNet weights: {e}")
    else:
        print(f"Warning: UNet weights file does not exist: {weights_path}")

    print("Using randomly initialized UNet model")
    return model


# 使用UNet进行图像去噪
def apply_unet_denoising(image, unet_model):
    """
    Apply the UNet model for medical image denoising
    image: numpy array (uint8) in range [0, 255]
    returns: numpy array (uint8) in range [0, 255]
    """
    if unet_model is None:
        print("UNet model is not available, skipping denoising step")
        return image

    try:
        # Convert to PyTorch tensor
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Apply UNet denoising
        with torch.no_grad():
            denoised = unet_model(image_tensor)

        # Convert back to numpy array
        denoised = denoised.squeeze().cpu().numpy()
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)

        return denoised
    except Exception as e:
        print(f"UNet denoising failed: {e}")
        return image


def enhanced_preprocessing(image_path, unet_model=None):
    """Enhanced image preprocessing pipeline with UNet denoising option"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        # Ensure image is in RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Apply UNet denoising if enabled
        if unet_model is not None:
            # Convert to grayscale for UNet
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Apply UNet denoising
            denoised_gray = apply_unet_denoising(gray, unet_model)

            # Convert back to RGB by stacking the same channel
            image = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2RGB)

        # Gaussian blur
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Resize to target size
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)

        # Normalize
        return image.astype('float32') / 255.0

    except Exception as e:
        print(f"Preprocessing error:{e}")
        # Error recovery - use gray image as replacement
        image = cv2.imread(image_path)
        if image is None:
            image = np.ones((384, 384, 3), dtype=np.float32) * 0.5
        else:
            image = cv2.resize(image, (384, 384))
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.astype('float32') / 255.0


def visualize_unet_effect(image_path, output_dir, unet_model):
    """Visualize UNet denoising effect for sample images"""
    try:
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)

        # Read and preprocess without UNet
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # Apply UNet denoising
        unet_image = blurred.copy()
        if unet_model is not None:
            gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
            denoised_gray = apply_unet_denoising(gray, unet_model)
            unet_image = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2RGB)

        # Resize for display
        display_size = (384, 384)
        orig_display = cv2.resize(image, display_size)
        blur_display = cv2.resize(blurred, display_size)
        unet_display = cv2.resize(unet_image, display_size)

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
        plt.imshow(unet_display)
        plt.title('UNet Denoised')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'unet_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"UNet comparison visualization saved to: {os.path.join(output_dir, 'unet_comparison.png')}")

    except Exception as e:
        print(f"UNet visualization error: {e}")


class MammographyDataset(Dataset):
    """Mammography image dataset class"""

    def __init__(self, dataframe, directory, x_col, y_col=None, transform=None, unet_model=None):
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.transform = transform
        self.unet_model = unet_model

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
        image = enhanced_preprocessing(image_path, self.unet_model)

        if self.transform:
            image = self.transform(image)

        if self.y_col:
            label = self.class_to_idx[row[self.y_col]]
            return image, label
        else:
            return image, row[self.x_col]


def load_and_analyze_data(data_dir, unet_model=None):
    """Load and analyze dataset"""
    print("Loading data...")
    train_path = r"D:\desktop\w2\w2\mammography_images\Training_set.csv"  # should be adjusted manually
    test_path = r"D:\desktop\w2\w2\mammography_images\Testing_set.csv"  # should be adjusted manually

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Training or testing CSV file does not exist: {train_path}, {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Data loading completed, training set size: {len(train)}, test set size: {len(test)}")

    # Data integrity verification
    required_columns = {'filename', 'label'}
    assert required_columns.issubset(train.columns), "Training data is missing required columns"
    assert 'filename' in test.columns, "Test data is missing the filename column"

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

    return train, test, class_weights, unique_classes


def create_data_loaders(train_df, test_df, class_weights, data_dir, unet_model=None, image_size=(384, 384)):
    """Create data loaders"""
    print("Creating data loaders...")

    # Training data augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation/test data augmentation
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
    print(f"Dataset split completed, training set: {len(train_df)}, validation set: {len(val_df)}")

    # Create datasets
    train_dataset = MammographyDataset(
        train_df,
        os.path.join(data_dir, 'train'),
        'filename',
        'label',
        transform=train_transform,
        unet_model=unet_model
    )

    val_dataset = MammographyDataset(
        val_df,
        os.path.join(data_dir, 'train'),
        'filename',
        'label',
        transform=val_transform,
        unet_model=unet_model
    )

    test_dataset = MammographyDataset(
        test_df,
        os.path.join(data_dir, 'test'),
        'filename',
        None,
        transform=val_transform,
        unet_model=unet_model
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    print("Data loaders created")
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def build_advanced_model(num_classes, image_size=(384, 384)):
    """Build enhanced EfficientNet-B4 model"""
    print("Building the model...")
    from torchvision.models import EfficientNet_B4_Weights

    # Load pre-trained EfficientNet-B4 model
    base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    features = base_model.features
    print("Pre-trained model loaded")

    # Unfreeze more layers
    for param in features.parameters():
        param.requires_grad = False
    for param in features[-10:].parameters():  # Unfreeze more layers
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
        sample_input = torch.randn(1, 3, *image_size).to(device)
        features_dim = feature_extractor(sample_input).shape[1]
    print(f"Actual feature dimension: {features_dim}")

    # Build classifier head
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
    print(f"Total model parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")

    return full_model


def train_model(model, train_loader, val_loader, class_weights, unique_classes, epochs, patience=10, min_lr=1e-7,
                output_dir='.'):
    """Train the model"""
    print("Starting model training...")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience,
                                                      min_lr=min_lr)

    # Early stopping
    best_val_auc = 0.0
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

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_all_preds = []
        train_all_labels = []

        print("Training...")
        accumulation_steps = 2  # Gradient accumulation steps

        for i, (inputs, labels) in enumerate(train_loader):
            # Manage memory usage
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Gradient accumulation

            loss.backward()

            # Update parameters every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0) * accumulation_steps
            correct += predicted.eq(labels).sum().item() * accumulation_steps

            # Collect training predictions for precision and recall calculation
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())

            # Print progress
            if (i + 1) % 20 == 0:
                print(
                    f"Processing batch {i + 1}/{len(train_loader)}, Current batch loss: {loss.item() * accumulation_steps:.4f}")

            # Clean cache periodically
            if (i + 1) % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        # Calculate training precision and recall
        train_micro_precision = precision_score(train_all_labels, train_all_preds, average='micro')
        train_micro_recall = recall_score(train_all_labels, train_all_preds, average='micro')

        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_accuracy = 100. * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['train_precision'].append(train_micro_precision)
        history['train_recall'].append(train_micro_recall)
        print(f"Training loss: {epoch_loss:.4f}, Training accuracy: {epoch_accuracy:.2f}%")
        print(f"Training precision: {train_micro_precision:.4f}, Training recall: {train_micro_recall:.4f}")

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
        print(f"Validation loss: {epoch_val_loss:.4f}, Validation accuracy: {epoch_val_accuracy:.2f}%")
        print(f"Validation precision: {val_micro_precision:.4f}, Validation recall: {val_micro_recall:.4f}")

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
        remaining_time = (epochs - epoch - 1) * avg_epoch_time
        print(f"Epoch time: {epoch_time:.2f}s, Average: {avg_epoch_time:.2f}s, Remaining: {remaining_time / 60:.1f} minutes")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            counter = 0
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved to {model_path}, Validation AUC: {best_val_auc:.4f}")
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered, Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
                break

        # Clean cache after each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    total_time = time.time() - start_time
    print(f"Training completed, Total time: {total_time / 60:.2f} minutes")

    # Save training time
    time_info = {
        'total_time': total_time,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'epoch_times': epoch_times
    }
    with open(os.path.join(output_dir, 'training_time.json'), 'w') as f:
        json.dump(time_info, f)

    return history


def evaluate_model(model, test_loader, class_to_idx, output_dir, unique_classes, device):
    """Evaluate the model and generate detailed report"""
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
                results.append({
                    'filename': filename,
                    'prediction': pred_class,
                    'probabilities': probs[i].tolist()
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
    print(f"Evaluation completed, Results saved to {results_path}")

    return results


def visualize_results(history, output_dir):
    """Enhanced training metrics visualization"""
    plt.figure(figsize=(18, 12))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Accuracy curve
    plt.subplot(2, 2, 1)
    if 'train_accuracy' in history and 'val_accuracy' in history:
        plt.plot(history['train_accuracy'], label='Training Accuracy', marker='o')
        plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
        plt.title('Accuracy over Epochs with UNet')
        plt.ylabel('Accuracy (%)')
    else:
        plt.text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Accuracy Data Missing')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Loss curve
    plt.subplot(2, 2, 2)
    if 'train_loss' in history and 'val_loss' in history:
        plt.plot(history['train_loss'], label='Training Loss', marker='o')
        plt.plot(history['val_loss'], label='Validation Loss', marker='s')
        plt.title('Loss over Epochs with UNet')
        plt.ylabel('Loss')
    else:
        plt.text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Loss Data Missing')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. AUC curve
    plt.subplot(2, 2, 3)
    if 'val_auc' in history:
        if 'train_auc' in history:
            plt.plot(history['train_auc'], label='Training AUC', marker='o')
        plt.plot(history['val_auc'], label='Validation AUC', marker='s')
        plt.title('AUC over Epochs with UNet')
        plt.ylabel('AUC')
    else:
        plt.text(0.5, 0.5, 'AUC data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('AUC Data Missing')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. Precision and recall curve
    plt.subplot(2, 2, 4)
    if 'train_precision' in history and 'val_precision' in history and 'train_recall' in history and 'val_recall' in history:
        plt.plot(history['train_precision'], label='Training Precision', marker='o')
        plt.plot(history['val_precision'], label='Validation Precision', marker='s')
        plt.plot(history['train_recall'], label='Training Recall', marker='^')
        plt.plot(history['val_recall'], label='Validation Recall', marker='v')
        plt.title('Precision and Recall over Epochs with UNet')
        plt.ylabel('Metric Value')
    else:
        plt.text(0.5, 0.5, 'Precision/Recall data not available', ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.title('Precision/Recall Data Missing')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save figure
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'UNet_training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training metrics visualization saved to: {plot_path}")


def main():
    # Print device information
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    # Get training parameters
    print("Please enter the number of training epochs (recommended 50):")
    epochs = int(input())

    print("Please enter the early stopping patience value (recommended 5 or 10):")
    patience = int(input())

    # Create argument parser
    parser = argparse.ArgumentParser(description='Mammography image classification model')
    parser.add_argument('--data_dir', type=str, default=r"D:\desktop\w3\w3\mammography_images", help='Data directory')      # should be adjusted manually
    parser.add_argument('--output_dir', type=str,
                        default=r"C:\Users\27785\PycharmProjects\PythonProject1\.venv\乳腺癌w5\output",        # should be adjusted manually
                        help='Output directory')
    parser.add_argument('--use_unet', action='store_true', help='Enable UNet denoising')
    parser.add_argument('--unet_weights', type=str, default="unet_denoiser.pth", help='UNet weights path')
    args = parser.parse_args()

    try:
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Load UNet model (if enabled)
        unet_model = None
        if args.use_unet:
            print("Loading UNet denoising model...")
            unet_model = load_unet_model(args.unet_weights)
            print("UNet model loaded")
        else:
            print("UNet denoising is not enabled")

        # Load data
        train_df, test_df, class_weights, unique_classes = load_and_analyze_data(args.data_dir, unet_model)

        # Create data loaders
        train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
            train_df, test_df, class_weights, args.data_dir, unet_model
        )

        # Build the model
        model = build_advanced_model(len(class_to_idx))

        # Train the model
        history = train_model(
            model,
            train_loader,
            val_loader,
            class_weights,
            unique_classes,
            epochs=epochs,
            patience=patience,
            output_dir=args.output_dir
        )

        # Visualize results
        visualize_results(history, args.output_dir)

        # Load the best model for evaluation
        best_model = build_advanced_model(len(class_to_idx))
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            best_model.load_state_dict(torch.load(best_model_path))
            best_model.to(device).eval()
            print(f"Loaded the best model from {best_model_path}")

            # Evaluate the model
            evaluate_model(best_model, test_loader, class_to_idx, args.output_dir, unique_classes, device)
        else:
            print(f"Warning: The best model file does not exist at {best_model_path}")

        # Save the final model
        final_model_path = os.path.join(args.output_dir, 'final_advanced_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Model saved as {final_model_path}")

        # Visualize UNet effect
        if unet_model is not None:
            sample_image = os.path.join(args.data_dir, 'train', train_df.iloc[0]['filename'])
            visualize_unet_effect(sample_image, args.output_dir, unet_model)

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()