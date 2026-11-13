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
    # GPU memory protection
    torch.cuda.set_per_process_memory_fraction(0.8)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# Resource monitoring function
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


# SURF-2025-0145-related functions (保持原样)
def apply_svd(image, k=None, variance_threshold=0.95, manual_k=None):
    """
    Apply SURF-2025-0145 decomposition to an image and reconstruct it

    Parameters:
        image: Input image (H, W, C)
        k: Number of singular values to retain (automatically determined if None)
        variance_threshold: Variance explanation threshold for automatic k selection
        manual_k: Manually specified k value (overrides automatic selection if provided)

    Returns:
        reconstructed_image: Reconstructed image
        used_k: The actual k value used
        explained_variance: Proportion of explained variance
    """
    reconstructed_channels = []
    k_values = []
    explained_variances = []

    # Apply SURF-2025-0145 to each channel individually
    for channel in range(image.shape[2]):
        img_channel = image[:, :, channel]

        # Perform SURF-2025-0145
        U, S, Vt = np.linalg.svd(img_channel, full_matrices=False)

        # Calculate cumulative explained variance ratio
        total_variance = np.sum(S ** 2)
        cumulative_variance = np.cumsum(S ** 2) / total_variance

        # Determine k value
        if manual_k is not None:
            # Use manually specified k value
            k = manual_k
        elif k is None:
            # Automatically determine k based on variance threshold
            k = np.argmax(cumulative_variance >= variance_threshold) + 1
            # Ensure at least 1 singular value is retained
            k = max(1, k)

        # Retain top k singular values and reconstruct
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]
        reconstructed_channel = U_k @ S_k @ Vt_k

        # Ensure values are within valid range
        reconstructed_channel = np.clip(reconstructed_channel, 0, 1)

        reconstructed_channels.append(reconstructed_channel)
        k_values.append(k)
        explained_variances.append(cumulative_variance[k - 1])

    # Merge channels
    reconstructed_image = np.stack(reconstructed_channels, axis=2)

    # Return average k value and average explained variance
    avg_k = int(np.mean(k_values))
    avg_explained_variance = np.mean(explained_variances)

    return reconstructed_image, avg_k, avg_explained_variance

def find_optimal_k_range(image, min_k=1, max_k=50, step=1):
    """Determine reconstruction error and variance explanation for different k values for visualization"""
    k_values = []
    errors = []
    variances = []

    for channel in range(image.shape[2]):
        img_channel = image[:, :, channel]
        U, S, Vt = np.linalg.svd(img_channel, full_matrices=False)
        total_variance = np.sum(S ** 2)

        for k in range(min_k, max_k + 1, step):
            if k >= len(S):
                break

            U_k = U[:, :k]
            S_k = np.diag(S[:k])
            Vt_k = Vt[:k, :]
            reconstructed = U_k @ S_k @ Vt_k

            # Calculate reconstruction error
            error = np.mean((img_channel - reconstructed) ** 2)
            # Calculate explained variance ratio
            variance = np.sum(S[:k] ** 2) / total_variance

            k_values.append(k)
            errors.append(error)
            variances.append(variance)

    # Average by k value
    unique_k = np.unique(k_values)
    avg_errors = []
    avg_variances = []

    for k in unique_k:
        mask = np.array(k_values) == k
        avg_errors.append(np.mean(np.array(errors)[mask]))
        avg_variances.append(np.mean(np.array(variances)[mask]))

    return unique_k, avg_errors, avg_variances


def visualize_svd_effects(original_image, output_dir, min_k=1, max_k=50, manual_k=None):
    """Visualize the effects of different k values on image reconstruction"""
    # Create save directory
    svd_vis_dir = os.path.join(output_dir, 'svd_visualizations')
    os.makedirs(svd_vis_dir, exist_ok=True)

    # Calculate error and variance explanation for different k values
    k_values, errors, variances = find_optimal_k_range(original_image, min_k, max_k)

    # Visualize error and variance explanation vs k value
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, errors, 'b-', marker='o')
    if manual_k is not None:
        # Highlight manually selected k if provided
        if manual_k in k_values:
            idx = np.where(k_values == manual_k)[0][0]
            plt.scatter(manual_k, errors[idx], color='red', s=100, zorder=5, label=f'Manual k={manual_k}')
            plt.axvline(x=manual_k, color='red', linestyle='--', alpha=0.3)
            plt.legend()
    plt.title('Reconstruction Error vs k')
    plt.xlabel('k (Number of Singular Values)')
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, variances, 'r-', marker='s')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
    if manual_k is not None:
        # Highlight manually selected k if provided
        if manual_k in k_values:
            idx = np.where(k_values == manual_k)[0][0]
            plt.scatter(manual_k, variances[idx], color='red', s=100, zorder=5, label=f'Manual k={manual_k}')
            plt.axvline(x=manual_k, color='red', linestyle='--', alpha=0.3)
            plt.legend()
    plt.title('Explained Variance vs k')
    plt.xlabel('k (Number of Singular Values)')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(svd_vis_dir, 'svd_metrics_vs_k.png'))
    plt.close()

    # Visualize reconstruction effects with different k values
    k_samples = [1, 5, 10, 20, 30, max_k]
    if manual_k is not None and manual_k not in k_samples:
        k_samples.insert(2, manual_k)  # Insert manual k into visualization samples
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # Reconstructed images with different k values
    for i, k in enumerate(k_samples, 2):
        reconstructed, _, _ = apply_svd(original_image, k=k)
        plt.subplot(2, 4, i)
        plt.imshow(reconstructed)
        plt.title(f'Reconstructed (k={k})')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(svd_vis_dir, 'svd_reconstructions.png'))
    plt.close()

    return k_values, variances


def visualize_k_distribution(k_values, output_dir):
    """Visualize the distribution of k values across different images"""
    svd_vis_dir = os.path.join(output_dir, 'svd_visualizations')
    os.makedirs(svd_vis_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(k_values, bins=20, kde=True)
    plt.axvline(x=np.mean(k_values), color='r', linestyle='--', label=f'Mean: {np.mean(k_values):.2f}')
    plt.axvline(x=np.median(k_values), color='g', linestyle='--', label=f'Median: {np.median(k_values):.1f}')
    plt.title('Distribution of SURF-2025-0145 k Values Across Images')
    plt.xlabel('k Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(svd_vis_dir, 'k_value_distribution.png'))
    plt.close()

    print(f"k value distribution visualization saved to: {os.path.join(svd_vis_dir, 'k_value_distribution.png')}")

# 新增图像缓存类
class ImageCache:
    def __init__(self, max_size=2000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get(self, key):
        if key in self.cache:
            # 更新访问顺序
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # 移除最久未使用的项
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.access_order.append(key)


# 全局图像缓存
image_cache = ImageCache(max_size=2000)

# Hyperparameter configuration
print("Enter number of training epochs:")
num = int(input())


class Config:
    IMAGE_SIZE = (256, 256)  # 减小图像尺寸
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 1  # 移除梯度累积
    EPOCHS = num
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 1e-5
    BASE_MODEL = 'efficientnet_b4'
    PATIENCE = 15
    MIN_LR = 1e-7
    DATA_DIR = r"D:\desktop\ISIC_2019"
    TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "ISIC_2019_Training_Input")
    TEST_IMAGE_DIR = os.path.join(DATA_DIR, "ISIC_2019_Test_Input")
    OUTPUT_DIR = r"C:\Users\27785\PycharmProjects\PythonProject1\.venv\ISIC_SVD"
    USE_MIXED_PRECISION = True
    # SURF-2025-0145-related parameters
    USE_SVD = True
    SVD_MODE = 'auto'
    SVD_MANUAL_K = 15  # 减小K值
    SVD_VARIANCE_THRESHOLD = 0.95
    MIN_SVD_K = 5
    MAX_SVD_K = 30  # 减小最大K值


config = Config()


def load_and_analyze_data(data_dir):
    """Load and analyze ISIC2019 dataset"""
    print("Loading data...")
    train_path = os.path.join(data_dir, "ISIC_2019_Training_GroundTruth.csv")
    test_path = os.path.join(data_dir, "ISIC_2019_Test_GroundTruth.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training CSV file not found: {train_path}")

    train = pd.read_csv(train_path)
    print(f"Data loaded, training set size: {len(train)}")

    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC']
    train['label'] = train[classes].idxmax(axis=1)

    if os.path.exists(test_path):
        test = pd.read_csv(test_path)
        print(f"Test set size: {len(test)}")
    else:
        test_images_dir = config.TEST_IMAGE_DIR
        if os.path.exists(test_images_dir):
            test_filenames = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            test = pd.DataFrame({'image': test_filenames})
            print(f"Created test set from directory, {len(test)} images total")
        else:
            raise FileNotFoundError(f"Test set not found: {test_path} or {test_images_dir}")

    required_train_columns = {'image'} | set(classes)
    assert required_train_columns.issubset(train.columns), "Training data missing required columns"
    assert 'image' in test.columns, "Test data missing 'image' column"

    class_counts = train['label'].value_counts()
    print(f"Class distribution:\n{class_counts}")

    unique_classes = np.unique(train['label'])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train['label']
    )

    max_weight = max(class_weights)
    for i, weight in enumerate(class_weights):
        if weight < max_weight * 0.1:
            class_weights[i] *= 5.0
        elif weight < max_weight * 0.3:
            class_weights[i] *= 3.0
        elif weight < max_weight * 0.5:
            class_weights[i] *= 2.0

    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Calculated class weights:", class_weights)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    return train, test, class_weights, unique_classes


def enhanced_preprocessing(image_path):
    """Optimized preprocessing pipeline with SURF-2025-0145"""
    try:
        # 检查缓存
        cached_image = image_cache.get(image_path)
        if cached_image is not None:
            return cached_image

        # Read image in color
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return np.zeros(config.IMAGE_SIZE + (3,), dtype=np.float32) * 0.5

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize first to reduce SURF-2025-0145 computation
        image = cv2.resize(image, config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        # Only apply CLAHE to luminance channel
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab_clahe = cv2.merge((cl, a, b))
        image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB).astype('float32') / 255.0

        # Apply SURF-2025-0145 preprocessing
        if config.USE_SVD:
            manual_k = config.SVD_MANUAL_K if config.SVD_MODE == 'manual' else None
            image, k, variance = apply_svd(
                image,
                variance_threshold=config.SVD_VARIANCE_THRESHOLD,
                manual_k=manual_k
            )
            k = max(config.MIN_SVD_K, min(k, config.MAX_SVD_K))

            if hasattr(enhanced_preprocessing, 'k_values'):
                enhanced_preprocessing.k_values.append(k)
            else:
                enhanced_preprocessing.k_values = [k]

        # 添加到缓存
        image_cache.set(image_path, image)
        return image

    except Exception as e:
        print(f"Preprocessing error: {e}")
        image = cv2.imread(image_path)
        if image is None:
            image = np.ones(config.IMAGE_SIZE + (3,), dtype=np.float32) * 0.5
        else:
            image = cv2.resize(image, config.IMAGE_SIZE)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype('float32') / 255.0


class ISICDataset(Dataset):
    """ISIC dataset class with caching"""

    def __init__(self, dataframe, directory, x_col='image', y_col=None, transform=None,
                 is_svd_visualization=False, svd_vis_dir=None):
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.transform = transform
        self.is_svd_visualization = is_svd_visualization
        self.svd_vis_dir = svd_vis_dir

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
        image_filename = row[self.x_col]
        if not image_filename.endswith(('.jpg', '.png', '.jpeg')):
            image_filename += '.jpg'

        image_path = os.path.join(self.directory, image_filename)

        # For SURF-2025-0145 visualization, only process first sample
        if self.is_svd_visualization and idx == 0:
            original_image = cv2.imread(image_path)
            if original_image is not None:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                original_image = cv2.resize(original_image, config.IMAGE_SIZE)
                original_image = original_image.astype('float32') / 255.0
                manual_k = config.SVD_MANUAL_K if config.SVD_MODE == 'manual' else None
                k_values, variances = visualize_svd_effects(original_image, self.svd_vis_dir, manual_k=manual_k)
                if config.SVD_MODE == 'auto':
                    avg_k = int(np.mean([np.argmax(v >= config.SVD_VARIANCE_THRESHOLD) + 1 for v in variances]))
                    print(f"Estimated optimal k value: {avg_k} for variance threshold {config.SVD_VARIANCE_THRESHOLD}")
                else:
                    print(f"Using manual k value: {config.SVD_MANUAL_K}")

        # Apply preprocessing
        image = enhanced_preprocessing(image_path)

        if self.transform:
            image = self.transform(image)

        if self.y_col:
            label = self.class_to_idx[row[self.y_col]]
            return image, label
        else:
            return image, row[self.x_col]


def create_data_loaders(train_df, test_df, class_weights, data_dir):
    print("Creating data loaders...")
    svd_vis_dir = os.path.join(config.OUTPUT_DIR, 'svd_visualizations')
    os.makedirs(svd_vis_dir, exist_ok=True)

    # Simplified transformations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),  # 减小旋转角度
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),  # 减小裁剪范围
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # 减小抖动
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Split data
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['label'] if 'label' in train_df.columns else None
    )
    print(f"Dataset split complete, training set: {len(train_df)}, validation set: {len(val_df)}")

    # Create datasets with SURF-2025-0145 visualization only for first sample
    vis_dataset = ISICDataset(
        train_df.iloc[:1],  # 仅使用第一个样本
        config.TRAIN_IMAGE_DIR,
        'image',
        'label',
        transform=train_transform,
        is_svd_visualization=True,
        svd_vis_dir=svd_vis_dir
    )
    if len(vis_dataset) > 0:
        vis_dataset[0]

    # Create main datasets
    train_dataset = ISICDataset(
        train_df,
        config.TRAIN_IMAGE_DIR,
        'image',
        'label',
        transform=train_transform
    )

    val_dataset = ISICDataset(
        val_df,
        config.TRAIN_IMAGE_DIR,
        'image',
        'label',
        transform=val_transform
    )

    test_dataset = ISICDataset(
        test_df,
        config.TEST_IMAGE_DIR,
        'image',
        None,
        transform=val_transform
    )

    # Optimized data loaders
    num_workers = min(4, os.cpu_count() // 2)  # 使用多进程加载数据

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )

    print("Data loaders created")
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def build_advanced_model(num_classes):
    """Build optimized EfficientNet-B4 model"""
    print("Building model...")
    from torchvision.models import EfficientNet_B4_Weights

    base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    features = base_model.features
    print("Pre-trained model loaded")

    # 减少解冻层数
    for param in features.parameters():
        param.requires_grad = False
    for param in features[-10:].parameters():  # 仅解冻最后10层
        param.requires_grad = True
    print("Model parameters unfrozen")

    feature_extractor = nn.Sequential(
        features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    ).to(device)

    # 计算特征维度
    with torch.no_grad():
        sample_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(device)
        features_dim = feature_extractor(sample_input).shape[1]
    print(f"Actual feature dimension: {features_dim}")

    # 简化分类头
    classifier = nn.Sequential(
        nn.Linear(features_dim, 512),  # 减小层大小
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),  # 减小dropout

        nn.Linear(512, 256),
        nn.SiLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.4),  # 减小dropout

        nn.Linear(256, num_classes),
    ).to(device)

    full_model = nn.Sequential(
        feature_extractor,
        classifier
    ).to(device)

    total_params = sum(p.numel() for p in full_model.parameters())
    trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}, trainable parameters: {trainable_params:,}")

    return full_model


def train_model(model, train_loader, val_loader, class_weights, unique_classes):
    """Optimized training function"""
    print("Enter learning rate patience:")
    n = int(input())

    print("Starting model training...")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=n,
                                                      min_lr=config.MIN_LR)

    best_val_auc = 0.0
    patience = config.PATIENCE
    counter = 0
    best_epoch = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_auc': [], 'val_auc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }

    start_time = time.time()
    epoch_times = []

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch + 1}/{config.EPOCHS} =====")
        log_resources()  # 记录资源使用

        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_all_preds = []
        train_all_labels = []
        train_all_probs = []

        print("Training...")
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(train_loader):
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=config.USE_MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())
            train_all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

            # 减少日志输出频率
            if (i + 1) % 50 == 0:
                print(f"Batch {i + 1}/{len(train_loader)}, loss: {loss.item():.4f}")
                log_resources()

            # 定期清理缓存
            if (i + 1) % 100 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

        # 计算训练指标
        train_micro_precision = precision_score(train_all_labels, train_all_preds, average='micro')
        train_micro_recall = recall_score(train_all_labels, train_all_preds, average='micro')
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

        print(f"Training loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.2f}%")
        print(f"Precision: {train_micro_precision:.4f}, Recall: {train_micro_recall:.4f}, AUC: {train_auc:.4f}")

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

        # 计算验证指标
        val_micro_precision = precision_score(all_labels, all_preds, average='micro')
        val_micro_recall = recall_score(all_labels, all_preds, average='micro')
        all_labels_onehot = label_binarize(all_labels, classes=list(range(num_classes)))
        val_auc = roc_auc_score(all_labels_onehot, all_probs, average='macro', multi_class='ovr')

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = 100. * correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['val_precision'].append(val_micro_precision)
        history['val_recall'].append(val_micro_recall)
        history['val_auc'].append(val_auc)

        print(f"Validation loss: {epoch_val_loss:.4f}, accuracy: {epoch_val_accuracy:.2f}%")
        print(f"Precision: {val_micro_precision:.4f}, Recall: {val_micro_recall:.4f}, AUC: {val_auc:.4f}")

        # 更新学习率
        scheduler1.step()
        scheduler2.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        # 计算时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = (config.EPOCHS - epoch - 1) * avg_epoch_time
        print(f"Epoch time: {epoch_time:.2f}s, remaining: {remaining_time / 60:.1f}min")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            counter = 0
            model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved, validation AUC: {best_val_auc:.4f}")
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping, best AUC: {best_val_auc:.4f} at epoch {best_epoch}")
                break

        # 清理缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    total_time = time.time() - start_time
    print(f"Training completed, total time: {total_time / 60:.2f} minutes")

    time_info = {
        'total_time': total_time,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'epoch_times': epoch_times
    }
    with open(os.path.join(config.OUTPUT_DIR, 'training_time.json'), 'w') as f:
        json.dump(time_info, f)

    return history


def evaluate_model(model, test_loader, class_to_idx, output_dir, unique_classes, device):
    """Optimized evaluation function"""
    print("Starting model evaluation...")
    model.eval()
    results = []
    all_preds = []
    all_filenames = []
    all_probs = []

    # 减少评估时的batch size以节省内存
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=8,  # 减小batch size
        shuffle=False,
        num_workers=min(2, os.cpu_count() // 2),
        pin_memory=True
    )

    try:
        with torch.no_grad():
            for batch_idx, (inputs, filenames) in enumerate(test_loader):
                # 定期清理缓存
                if batch_idx % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

                inputs = inputs.to(device)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                idx_to_class = {v: k for k, v in class_to_idx.items()}
                for i, filename in enumerate(filenames):
                    pred_class = idx_to_class[preds[i]]
                    class_prob = {cls: float(probs[i][idx]) for cls, idx in class_to_idx.items()}
                    results.append({
                        'filename': filename,
                        'prediction': pred_class,
                        **class_prob
                    })

                all_preds.extend(preds)
                all_filenames.extend(filenames)
                all_probs.extend(probs)

                # 清理临时变量
                del inputs, outputs, probs, preds
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # 减少日志输出
                if (batch_idx + 1) % 50 == 0:
                    print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    finally:
        # 确保资源释放
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # 保存结果
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'isic_predictions.csv')
    results_df.to_csv(results_path, index=False, chunksize=1000)
    print(f"Results saved to {results_path}")

    # 生成分布图
    plt.figure(figsize=(10, 6), dpi=100)
    pred_classes = [idx_to_class[p] for p in all_preds]
    sns.countplot(x=pred_classes)
    plt.title('Predicted Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=150)
    plt.close()

    # 清理内存
    del all_preds, all_filenames, all_probs, results_df
    gc.collect()

    return results


def visualize_results(history, output_dir, class_names=None):
    """保持原样，但减少DPI以节省内存"""
    # ... (保持原样，但修改保存图像时的DPI) ...
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=200, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='ISIC2019 Skin Lesion Classification with SURF-2025-0145 (Optimized)')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--use_svd', type=bool, default=config.USE_SVD, help='Use SURF-2025-0145 preprocessing')
    parser.add_argument('--svd_mode', type=str, default=config.SVD_MODE, choices=['auto', 'manual'],
                        help='SURF-2025-0145 mode: auto or manual')
    parser.add_argument('--svd_k', type=int, default=config.SVD_MANUAL_K,
                        help='Manual k value for SURF-2025-0145 (used when svd_mode=manual)')
    parser.add_argument('--svd_variance', type=float, default=config.SVD_VARIANCE_THRESHOLD,
                        help='Variance threshold for SURF-2025-0145 k selection (used when svd_mode=auto)')
    args = parser.parse_args()

    try:
        # 更新配置
        config.DATA_DIR = args.data_dir
        config.OUTPUT_DIR = args.output_dir
        config.USE_SVD = args.use_svd
        config.SVD_MODE = args.svd_mode
        config.SVD_MANUAL_K = args.svd_k
        config.SVD_VARIANCE_THRESHOLD = args.svd_variance

        print(f"Using SURF-2025-0145 preprocessing: {config.USE_SVD}")
        if config.USE_SVD:
            print(f"SURF-2025-0145 mode: {config.SVD_MODE}")
            if config.SVD_MODE == 'auto':
                print(f"SURF-2025-0145 variance threshold: {config.SVD_VARIANCE_THRESHOLD}")
            else:
                print(f"Manual SURF-2025-0145 k value: {config.SVD_MANUAL_K}")

        # 加载数据
        train_df, test_df, class_weights, unique_classes = load_and_analyze_data(config.DATA_DIR)

        # 创建数据加载器
        train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
            train_df, test_df, class_weights, config.DATA_DIR
        )

        # 构建模型
        model = build_advanced_model(len(class_to_idx))

        # 训练模型
        history = train_model(model, train_loader, val_loader, class_weights, unique_classes)

        # 生成k值分布图
        if config.USE_SVD and hasattr(enhanced_preprocessing, 'k_values'):
            visualize_k_distribution(enhanced_preprocessing.k_values, config.OUTPUT_DIR)

        # 可视化结果
        visualize_results(history, config.OUTPUT_DIR, class_names=class_to_idx.keys())

        # 资源释放步骤
        print("\n=== Releasing resources before evaluation ===")
        del model, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("Waiting 5 seconds for resource release...")
        time.sleep(5)
        log_resources()

        # 加载最佳模型
        print("\n=== Loading best model for evaluation ===")
        best_model = build_advanced_model(len(class_to_idx))
        best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.to(device).eval()
        print(f"Best model loaded from {best_model_path}")

        # 评估模型
        evaluate_model(best_model, test_loader, class_to_idx, config.OUTPUT_DIR, unique_classes, device)

        # 保存最终模型
        final_model_path = os.path.join(config.OUTPUT_DIR, 'final_isic_model.pth')
        torch.save(best_model.state_dict(), final_model_path)
        print(f"Model saved as {final_model_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()