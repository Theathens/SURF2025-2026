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
import glob  # 新增：用于检查文件存在性
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

# Detect and set device - modified for Apple MPS support
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # GPU memory protection - limit maximum memory usage
    torch.cuda.set_per_process_memory_fraction(0.8)  # Only use 80% of available memory
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
elif device.type == "mps":
    print(f"MPS acceleration enabled")
    # MPS memory settings
    os.environ["PYTORCH_MPS_ALLOC_CONF"] = "max_split_size_mb:32"


# Enhanced SURF-2025-0145 with automatic k adjustment
def apply_svd(image, variance_threshold=0.95, min_k=5, max_k=50, adaptive_threshold=False):
    """
    Apply SURF-2025-0145 decomposition to an image with enhanced automatic k adjustment

    Parameters:
        image: Input image (H, W, C)
        variance_threshold: Base variance explanation threshold for automatic k selection
        min_k: Minimum allowed k value
        max_k: Maximum allowed k value
        adaptive_threshold: Whether to use content-based adaptive thresholding

    Returns:
        reconstructed_image: Reconstructed image
        used_k: The actual k value used
        explained_variance: Proportion of explained variance
    """
    reconstructed_channels = []
    k_values = []
    explained_variances = []

    # Calculate image complexity for adaptive thresholding
    if adaptive_threshold:
        gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        entropy = cv2.filter2D(gray_image, -1, np.ones((3,3))/9)
        complexity = np.mean(entropy) / 255.0

        # Adjust threshold based on image complexity (more complex images need higher k)
        adjusted_threshold = variance_threshold + (1 - variance_threshold) * min(complexity, 0.7)
        adjusted_threshold = min(adjusted_threshold, 0.99)  # Cap at 99%
    else:
        adjusted_threshold = variance_threshold

    # Apply SURF-2025-0145 to each channel individually
    for channel in range(image.shape[2]):
        img_channel = image[:, :, channel]

        # Perform SURF-2025-0145
        U, S, Vt = np.linalg.svd(img_channel, full_matrices=False)

        # Calculate cumulative explained variance ratio
        total_variance = np.sum(S **2)
        cumulative_variance = np.cumsum(S** 2) / total_variance

        # Determine k value with constraints
        k = np.argmax(cumulative_variance >= adjusted_threshold) + 1
        k = max(min_k, min(k, max_k))  # Clamp k within valid range

        # Retain top k singular values and reconstruct
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]
        reconstructed_channel = U_k @ S_k @ Vt_k

        # Ensure values are within valid range
        reconstructed_channel = np.clip(reconstructed_channel, 0, 1)

        reconstructed_channels.append(reconstructed_channel)
        k_values.append(k)
        explained_variances.append(cumulative_variance[k-1])

    # Merge channels
    reconstructed_image = np.stack(reconstructed_channels, axis=2)

    # Return average k value and average explained variance
    avg_k = int(np.mean(k_values))
    avg_explained_variance = np.mean(explained_variances)

    return reconstructed_image, avg_k, avg_explained_variance, adjusted_threshold

def find_optimal_k_range(image, min_k=1, max_k=50, step=1):
    """Determine reconstruction error and variance explanation for different k values for visualization"""
    k_values = []
    errors = []
    variances = []

    for channel in range(image.shape[2]):
        img_channel = image[:, :, channel]
        U, S, Vt = np.linalg.svd(img_channel, full_matrices=False)
        total_variance = np.sum(S **2)

        for k in range(min_k, max_k + 1, step):
            if k >= len(S):
                break

            U_k = U[:, :k]
            S_k = np.diag(S[:k])
            Vt_k = Vt[:k, :]
            reconstructed = U_k @ S_k @ Vt_k

            # Calculate reconstruction error
            error = np.mean((img_channel - reconstructed)** 2)
            # Calculate explained variance ratio
            variance = np.sum(S[:k] **2) / total_variance

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

def visualize_svd_effects(original_image, output_dir, min_k=1, max_k=50):
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
    plt.title('Reconstruction Error vs k')
    plt.xlabel('k (Number of Singular Values)')
    plt.ylabel('MSE')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, variances, 'r-', marker='s')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
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
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # Reconstructed images with different k values
    for i, k in enumerate(k_samples, 2):
        # For visualization, create a temporary function call
        reconstructed = np.zeros_like(original_image)
        for channel in range(original_image.shape[2]):
            img_channel = original_image[:, :, channel]
            U, S, Vt = np.linalg.svd(img_channel, full_matrices=False)
            U_k = U[:, :k]
            S_k = np.diag(S[:k])
            Vt_k = Vt[:k, :]
            reconstructed[:, :, channel] = U_k @ S_k @ Vt_k
        reconstructed = np.clip(reconstructed, 0, 1)

        plt.subplot(2, 4, i)
        plt.imshow(reconstructed)
        plt.title(f'Reconstructed (k={k})')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(svd_vis_dir, 'svd_reconstructions.png'))
    plt.close()

    return k_values, variances

def visualize_k_analysis(k_values, thresholds, output_dir):
    """Enhanced visualization of k values distribution and relationship with thresholds"""
    svd_vis_dir = os.path.join(output_dir, 'svd_visualizations')
    os.makedirs(svd_vis_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Histogram of k values
    axes[0, 0].hist(k_values, bins=range(min(k_values), max(k_values) + 2), alpha=0.7, color='blue')
    axes[0, 0].axvline(x=np.mean(k_values), color='r', linestyle='--', label=f'Mean: {np.mean(k_values):.2f}')
    axes[0, 0].axvline(x=np.median(k_values), color='g', linestyle='--', label=f'Median: {np.median(k_values):.1f}')
    axes[0, 0].set_title('Distribution of SURF-2025-0145 k Values')
    axes[0, 0].set_xlabel('k Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    axes[0, 0].legend()

    # 2. KDE plot of k values
    sns.kdeplot(k_values, ax=axes[0, 1], fill=True, color='blue')
    axes[0, 1].axvline(x=np.mean(k_values), color='r', linestyle='--', label=f'Mean: {np.mean(k_values):.2f}')
    axes[0, 1].set_title('Density Distribution of k Values')
    axes[0, 1].set_xlabel('k Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    axes[0, 1].legend()

    # 3. Threshold distribution
    if thresholds:
        axes[1, 0].hist(thresholds, bins=20, alpha=0.7, color='green')
        axes[1, 0].axvline(x=np.mean(thresholds), color='r', linestyle='--', label=f'Mean: {np.mean(thresholds):.3f}')
        axes[1, 0].set_title('Distribution of Adaptive Thresholds')
        axes[1, 0].set_xlabel('Variance Threshold')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No threshold data available', ha='center', va='center', transform=axes[1,0].transAxes)

    # 4. Relationship between k and threshold
    if thresholds and len(k_values) == len(thresholds):
        axes[1, 1].scatter(thresholds, k_values, alpha=0.5, color='purple')
        z = np.polyfit(thresholds, k_values, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(thresholds, p(thresholds), "r--")
        axes[1, 1].set_title('k Value vs Variance Threshold')
        axes[1, 1].set_xlabel('Variance Threshold')
        axes[1, 1].set_ylabel('k Value')
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    else:
        axes[1, 1].text(0.5, 0.5, 'No relationship data available', ha='center', va='center', transform=axes[1,1].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(svd_vis_dir, 'k_value_analysis.png'))
    plt.close()

    print(f"k value analysis visualization saved to: {os.path.join(svd_vis_dir, 'k_value_analysis.png')}")


# Resource monitoring
def log_resources():
    if device.type == "cuda":
        alloc_mem = torch.cuda.memory_allocated() / (1024 **3)  # GB
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024** 3)
        print(f"GPU memory usage: {alloc_mem:.2f}/{total_mem:.2f} GB ({alloc_mem / total_mem:.1%})")
    elif device.type == "mps":
        print("MPS memory monitoring not available")

    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        print(f"CPU usage: {cpu_percent}% | RAM usage: {ram_percent}%")
    except ImportError:
        pass

# Hyperparameter configuration


class Config:
    IMAGE_SIZE = (384, 384)      # Adjusted to 384x384 for balance between accuracy and memory
    BATCH_SIZE = 16              # Batch size - reduced for MPS compatibility
    ACCUMULATION_STEPS = 2       # Gradient accumulation steps, equivalent to BATCH_SIZE=32
    EPOCHS = 1
    LEARNING_RATE = 3e-5         # Lower learning rate for more complex ISIC data
    WEIGHT_DECAY = 1e-5
    BASE_MODEL = 'efficientnet_b4'
    PATIENCE = 15                # More patience for ISIC training
    MIN_LR = 1e-7
    # 请根据实际情况修改以下路径
    DATA_DIR = r"D:\desktop\ISIC_2019"  # Update with your actual path
    OUTPUT_DIR = r"C:\Users\27785\PycharmProjects\PythonProject1\.venv\ISIC_SVD_RESULT"  # Update with your actual path
    TRAIN_IMAGE_SUBPATH = "ISIC_2019_Training_Input"  # 训练图像相对路径
    TEST_IMAGE_SUBPATH = "ISIC_2019_Test_Input"  # 测试图像相对路径
    USE_MIXED_PRECISION = True
    # SURF-2025-0145-related parameters
    USE_SVD = True               # Whether to use SURF-2025-0145 preprocessing
    SVD_ADAPTIVE_THRESHOLD = True  # Use content-based adaptive thresholding
    SVD_BASE_VARIANCE_THRESHOLD = 0.90  # Base variance threshold for adaptive adjustment
    MIN_SVD_K = 5                # Minimum k value limit
    MAX_SVD_K = 50               # Maximum k value limit


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
        'NV',   # Nevus
        'BCC',  # Basal cell carcinoma
        'AK',   # Actinic keratosis
        'BKL',  # Benign keratosis
        'DF',   # Dermatofibroma
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
        test_images_dir = os.path.join(data_dir, config.TEST_IMAGE_SUBPATH)
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
    """Enhanced preprocessing pipeline for skin lesion images with optional SURF-2025-0145"""
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path} (file may be corrupted)")

        # Convert to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        image = image.astype('float32') / 255.0

        # Apply SURF-2025-0145 preprocessing with auto k adjustment
        if config.USE_SVD:
            # Apply enhanced SURF-2025-0145 with automatic k adjustment
            image, k, variance, threshold = apply_svd(
                image,
                variance_threshold=config.SVD_BASE_VARIANCE_THRESHOLD,
                min_k=config.MIN_SVD_K,
                max_k=config.MAX_SVD_K,
                adaptive_threshold=config.SVD_ADAPTIVE_THRESHOLD
            )

            # Track k values and thresholds for analysis
            if hasattr(enhanced_preprocessing, 'k_values'):
                enhanced_preprocessing.k_values.append(k)
                enhanced_preprocessing.thresholds.append(threshold)
            else:
                enhanced_preprocessing.k_values = [k]
                enhanced_preprocessing.thresholds = [threshold]

        return image

    except Exception as e:
        print(f"Preprocessing error: {e}")
        # 错误恢复：创建空白图像作为替代
        return np.ones((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), dtype=np.float32) * 0.5

class ISICDataset(Dataset):
    """ISIC dataset class"""
    def __init__(self, dataframe, directory, x_col='image', y_col=None, transform=None,
                 is_svd_visualization=False, svd_vis_dir=None):
        self.dataframe = dataframe
        self.directory = directory  # 图像所在目录的完整路径
        self.x_col = x_col
        self.y_col = y_col
        self.transform = transform
        self.is_svd_visualization = is_svd_visualization  # 是否生成SVD可视化样本
        self.svd_vis_dir = svd_vis_dir

        # 存储SVD相关统计信息
        self.svd_k_values = []
        self.svd_variances = []

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
        # ISIC图像通常存储为.jpg或.png格式
        image_filename = row[self.x_col]

        # 尝试多种可能的文件扩展名
        possible_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        image_path = None

        # 检查带扩展名的文件是否存在
        for ext in possible_extensions:
            test_path = os.path.join(self.directory, f"{image_filename}{ext}")
            if os.path.exists(test_path):
                image_path = test_path
                break

        # 如果找不到带扩展名的，检查文件名是否已包含扩展名
        if image_path is None:
            test_path = os.path.join(self.directory, image_filename)
            if os.path.exists(test_path):
                image_path = test_path
            else:
                raise FileNotFoundError(
                    f"Image file not found for {image_filename} in {self.directory} "
                    f"with extensions {possible_extensions}"
                )

        # 用于SVD可视化，需要原始图像
        if self.is_svd_visualization and idx == 0:  # 只处理第一个样本用于可视化
            # 读取原始图像
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise FileNotFoundError(f"Cannot read image for visualization: {image_path}")

            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = cv2.resize(original_image, config.IMAGE_SIZE)
            original_image = original_image.astype('float32') / 255.0

            # 生成SVD可视化
            k_values, variances = visualize_svd_effects(original_image, self.svd_vis_dir)

            # 计算此样本的自动k值
            _, auto_k, _, _ = apply_svd(
                original_image,
                variance_threshold=config.SVD_BASE_VARIANCE_THRESHOLD,
                min_k=config.MIN_SVD_K,
                max_k=config.MAX_SVD_K,
                adaptive_threshold=config.SVD_ADAPTIVE_THRESHOLD
            )

            print(f"Automatic k value for visualization sample: {auto_k}")

        # 应用预处理
        image = enhanced_preprocessing(image_path)

        if self.transform:
            image = self.transform(image)

        if self.y_col:
            label = self.class_to_idx[row[self.y_col]]
            return image, label
        else:
            return image, row[self.x_col]

def create_data_loaders(train_df, test_df, class_weights, data_dir):
    """Create data loaders with corrected path handling"""
    print("Creating data loaders...")

    # 创建SVD可视化目录
    svd_vis_dir = os.path.join(config.OUTPUT_DIR, 'svd_visualizations')
    os.makedirs(svd_vis_dir, exist_ok=True)

    # 构建完整的训练和测试图像目录路径
    train_images_dir = os.path.join(data_dir, config.TRAIN_IMAGE_SUBPATH)
    test_images_dir = os.path.join(data_dir, config.TEST_IMAGE_SUBPATH)

    # 验证图像目录是否存在
    if not os.path.exists(train_images_dir):
        raise NotADirectoryError(f"Training images directory not found: {train_images_dir}")
    if not os.path.exists(test_images_dir):
        raise NotADirectoryError(f"Test images directory not found: {test_images_dir}")

    print(f"Using training images from: {train_images_dir}")
    print(f"Using test images from: {test_images_dir}")

    # 训练数据增强（针对皮肤病变图像调整）
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),  # 更大的旋转角度
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 更强的颜色抖动
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 验证/测试数据转换
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 将训练集拆分为训练集和验证集 - 确保分层抽样
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['label'] if 'label' in train_df.columns else None
    )
    print(f"Dataset split complete, training set: {len(train_df)}, validation set: {len(val_df)}")

    # 创建数据集，第一个用于生成SVD可视化
    vis_dataset = ISICDataset(
        train_df,
        train_images_dir,  # 使用正确的训练图像目录
        'image',
        'label',
        transform=train_transform,
        is_svd_visualization=True,
        svd_vis_dir=svd_vis_dir
    )
    # 触发第一个样本的加载以生成可视化
    if len(vis_dataset) > 0:
        vis_dataset[0]

    # 创建数据集
    train_dataset = ISICDataset(
        train_df,
        train_images_dir,  # 使用正确的训练图像目录
        'image',
        'label',
        transform=train_transform
    )

    val_dataset = ISICDataset(
        val_df,
        train_images_dir,  # 验证集也使用训练图像目录
        'image',
        'label',
        transform=val_transform
    )

    test_dataset = ISICDataset(
        test_df,
        test_images_dir,  # 使用正确的测试图像目录
        'image',
        None,
        transform=val_transform
    )

    # 根据设备调整num_workers
    num_workers = 0 if device.type in ['cuda', 'mps'] else 2

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False
    )

    print("Data loaders created")
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

def build_advanced_model(num_classes):
    """Build enhanced EfficientNet-B4 model (adjusted for multi-class classification)"""
    print("Building model...")
    from torchvision.models import EfficientNet_B4_Weights

    # 加载预训练的EfficientNet-B4模型
    base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    features = base_model.features
    print("Pre-trained model loaded")

    # 解冻更多层以适应复杂的ISIC特征
    for param in features.parameters():
        param.requires_grad = False
    # 对于ISIC数据，解冻更多层以捕获复杂的皮肤病变特征
    for param in features[-15:].parameters():  # 比原始多解冻5层
        param.requires_grad = True
    print("Model parameters unfrozen")

    # 构建特征提取器
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

    # 构建分类器头（用于7类分类任务）
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

    # 组合完整模型
    full_model = nn.Sequential(
        feature_extractor,
        classifier
    ).to(device)

    # 打印模型统计信息
    total_params = sum(p.numel() for p in full_model.parameters())
    trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}, trainable parameters: {trainable_params:,}")

    return full_model

def train_model(model, train_loader, val_loader, class_weights, unique_classes):
    """Train the model (adjusted for multi-class task)"""

    print("Enter learning rate patience:")
    n = int(input())

    print("Starting model training...")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # 学习率调度器
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=n, min_lr=config.MIN_LR)

    # 早停机制
    best_val_auc = 0.0
    patience = config.PATIENCE
    counter = 0
    best_epoch = 0

    # 训练历史
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

    # 记录开始时间
    start_time = time.time()
    epoch_times = []


    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch+1}/{config.EPOCHS} =====")

        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_all_preds = []
        train_all_labels = []
        train_all_probs = []

        print("Training...")
        for i, (inputs, labels) in enumerate(train_loader):
            # 管理内存使用
            if device.type in ['cuda', 'mps']:
                torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

            inputs, labels = inputs.to(device), labels.to(device)

            # MPS上尚未完全支持混合精度
            with torch.amp.autocast(device_type=device.type, enabled=config.USE_MIXED_PRECISION and device.type != 'mps'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / config.ACCUMULATION_STEPS  # 梯度累积

            loss.backward()

            # 每accumulation_steps更新一次参数
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * config.ACCUMULATION_STEPS * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0) * config.ACCUMULATION_STEPS
            correct += predicted.eq(labels).sum().item() * config.ACCUMULATION_STEPS

            # 收集训练预测用于指标计算
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())

            # 收集概率用于AUC计算
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            train_all_probs.extend(probs)

            # 打印进度
            if (i+1) % 20 == 0:
                print(f"Processing batch {i+1}/{len(train_loader)}, current batch loss: {loss.item()*config.ACCUMULATION_STEPS:.4f}")

            # 定期清理缓存
            if (i+1) % 50 == 0 and device.type in ['cuda', 'mps']:
                torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

        # 计算训练精度和召回率
        train_micro_precision = precision_score(train_all_labels, train_all_preds, average='micro')
        train_micro_recall = recall_score(train_all_labels, train_all_preds, average='micro')

        # 计算训练AUC（多类）
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

        # 验证阶段
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
                # 管理内存使用
                if device.type in ['cuda', 'mps']:
                    torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

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

        # 计算验证精度和召回率
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

        # 计算AUC（多类）
        num_classes = len(unique_classes)
        all_labels_onehot = label_binarize(all_labels, classes=list(range(num_classes)))
        val_auc = roc_auc_score(all_labels_onehot, all_probs, average='macro', multi_class='ovr')
        history['val_auc'].append(val_auc)
        print(f"Validation macro-average AUC: {val_auc:.4f}")

        # 更新学习率
        scheduler1.step()
        scheduler2.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = (config.EPOCHS - epoch - 1) * avg_epoch_time
        print(f"Epoch time: {epoch_time:.2f}s, average: {avg_epoch_time:.2f}s, remaining: {remaining_time / 60:.1f}min")

        # 保存最佳模型
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

        # 每个epoch后清理缓存
        if device.type in ['cuda', 'mps']:
            torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()
            gc.collect()

    total_time = time.time() - start_time
    print(f"Training completed, total time: {(time.time() - start_time)/60:.2f} minutes")

    # 保存训练时间信息
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
            # 管理内存使用
            if device.type in ['cuda', 'mps']:
                torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

            inputs = inputs.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=config.USE_MIXED_PRECISION and device.type != 'mps'):
                outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            idx_to_class = {v: k for k, v in class_to_idx.items()}
            for i, filename in enumerate(filenames):
                pred_class = idx_to_class[preds[i]]
                # 为ISIC结果添加每个类别的概率
                class_prob = {cls: float(probs[i][idx]) for cls, idx in class_to_idx.items()}
                results.append({
                    'filename': filename,
                    'prediction': pred_class,
                    **class_prob
                })

            all_preds.extend(preds)
            all_filenames.extend(filenames)
            all_probs.extend(probs)

            # 定期清理缓存
            if device.type in ['cuda', 'mps']:
                torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

    # 保存结果（ISIC格式）
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'isic_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation completed, results saved to {results_path}")

    # 生成类别分布直方图
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

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 准确率曲线
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

    # 2. 损失曲线
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

    # 3. AUC曲线
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

    # 4. 精确率和召回率曲线
    plt.subplot(2, 2, 4)
    if 'train_precision' in history and 'val_precision' in history and 'train_recall' in history and 'val_recall' in history:
        plt.plot(history['train_precision'], label='Training Precision', marker='o')
        plt.plot(history['val_precision'], label='Validation Precision', marker='s')
        plt.plot(history['train_recall'], label='Training Recall', marker='^')
        plt.plot(history['val_recall'], label='Validation Recall', marker='v')
        plt.title('Precision and Recall Over Epochs')
        plt.ylabel('Metric Value')
    else:
        plt.text(0.5, 0.5, 'Precision/Recall data not available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Precision/Recall Data Missing')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training metrics visualization saved to: {plot_path}")

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='ISIC2019 Skin Lesion Image Classification Model with SURF-2025-0145')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--use_svd', type=bool, default=config.USE_SVD, help='Whether to use SURF-2025-0145 preprocessing')
    parser.add_argument('--svd_adaptive', type=bool, default=config.SVD_ADAPTIVE_THRESHOLD,
                        help='Whether to use adaptive thresholding for SURF-2025-0145')
    parser.add_argument('--svd_base_threshold', type=float, default=config.SVD_BASE_VARIANCE_THRESHOLD,
                        help='Base variance threshold for SURF-2025-0145 k selection')
    parser.add_argument('--min_svd_k', type=int, default=config.MIN_SVD_K,
                        help='Minimum k value for SURF-2025-0145')
    parser.add_argument('--max_svd_k', type=int, default=config.MAX_SVD_K,
                        help='Maximum k value for SURF-2025-0145')
    args = parser.parse_args()

    try:
        # 更新配置
        config.DATA_DIR = args.data_dir
        config.OUTPUT_DIR = args.output_dir
        config.USE_SVD = args.use_svd
        config.SVD_ADAPTIVE_THRESHOLD = args.svd_adaptive
        config.SVD_BASE_VARIANCE_THRESHOLD = args.svd_base_threshold
        config.MIN_SVD_K = args.min_svd_k
        config.MAX_SVD_K = args.max_svd_k

        print(f"Using SURF-2025-0145 preprocessing: {config.USE_SVD}")
        if config.USE_SVD:
            print(f"SURF-2025-0145 adaptive thresholding: {config.SVD_ADAPTIVE_THRESHOLD}")
            print(f"SURF-2025-0145 base variance threshold: {config.SVD_BASE_VARIANCE_THRESHOLD}")
            print(f"SURF-2025-0145 k range: {config.MIN_SVD_K} - {config.MAX_SVD_K}")

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

        # 如果使用SVD，生成k值分析可视化
        if config.USE_SVD and hasattr(enhanced_preprocessing, 'k_values'):
            visualize_k_analysis(enhanced_preprocessing.k_values, enhanced_preprocessing.thresholds, config.OUTPUT_DIR)

        # 可视化结果
        visualize_results(history, config.OUTPUT_DIR, class_names=class_to_idx.keys())

        # 加载最佳模型进行评估
        best_model = build_advanced_model(len(class_to_idx))
        best_model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth')))
        best_model.to(device).eval()

        # 评估模型
        evaluate_model(best_model, test_loader, class_to_idx, config.OUTPUT_DIR, unique_classes, device)

        # 保存最终模型
        final_model_path = os.path.join(config.OUTPUT_DIR, 'final_isic_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Model saved as {final_model_path}")

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
