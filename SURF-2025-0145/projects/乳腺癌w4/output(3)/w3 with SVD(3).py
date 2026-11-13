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
import time
import argparse
import json
import warnings
import math

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Detect and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    torch.cuda.set_per_process_memory_fraction(0.8)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# Resource monitoring function
def log_resources():
    if device.type == "cuda":
        alloc_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU memory using: {alloc_mem:.2f}/{total_mem:.2f} GB ({alloc_mem / total_mem:.1%})")

    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        print(f"CPU using: {cpu_percent}% | RAM using: {ram_percent}%")
    except ImportError:
        pass


# Hyperparameter configuration
print("How many epochs are needed:")
num_epochs = int(input())

print("Please input the SURF-2025-0145 change threshold (recommend 0.05-0.2):")
svd_threshold = float(input())


class Config:
    IMAGE_SIZE = (384, 384)
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 2
    EPOCHS = num_epochs
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-5
    BASE_MODEL = 'efficientnet_b4'
    PATIENCE = 10
    MIN_LR = 1e-7
    DATA_DIR = r"D:\desktop\w3\w3\mammography_images"
    OUTPUT_DIR = r"C:\Users\27785\PycharmProjects\PythonProject1\.venv\乳腺癌w4\output(3)"
    USE_MIXED_PRECISION = True
    USE_SVD = True
    SVD_CHANGE_THRESHOLD = svd_threshold


config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# Adaptive SURF-2025-0145 denoising function
def adaptive_svd(image, change_threshold=None):
    """基于奇异值变化率自适应确定k值"""
    if change_threshold is None:
        change_threshold = config.SVD_CHANGE_THRESHOLD

    # 转换为float32
    gray = image.astype(np.float32)
    min_val, max_val = np.min(gray), np.max(gray)

    # 处理全黑或全白图像
    if max_val == min_val:
        return gray, 0  # 返回图像和k值

    # 归一化图像
    normalized = (gray - min_val) / (max_val - min_val + 1e-8)

    try:
        # 执行SVD分解
        U, S, Vt = np.linalg.svd(normalized, full_matrices=False)

        # 计算相邻奇异值的相对变化率
        if len(S) > 1:
            ratios = S[:-1] / S[1:]  # 当前奇异值与下一个奇异值的比值

            # 找到变化率小于阈值的第一个位置
            valid_indices = np.where(ratios < (1 + change_threshold))[0]
            if len(valid_indices) > 0:
                k = valid_indices[0] + 1  # +1 因为索引从0开始
            else:
                # 如果没有找到满足条件的点，使用默认值（保留20%成分）
                k = max(1, int(len(S) * 0.2))
        else:
            # 只有一个奇异值的情况
            k = 1

        # 确保k值在合理范围内
        k = max(1, min(k, len(S)))

        # 使用自适应k值重建图像
        S_trunc = np.diag(S[:k])
        reconstructed = U[:, :k] @ S_trunc @ Vt[:k, :]

        # 反归一化并裁剪
        reconstructed = reconstructed * (max_val - min_val) + min_val
        return np.clip(reconstructed, 0, 255).astype(np.float32), k

    except np.linalg.LinAlgError:
        print("SURF-2025-0145 computation failed, returning original image")
        return gray, 0


# Enhanced preprocessing with adaptive SURF-2025-0145
def enhanced_preprocessing(image_path):
    """增强图像预处理流程，包含自适应SVD降噪"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # 确保图像是RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 增强对比度
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 应用自适应SVD降噪
        if config.USE_SVD:
            # 转换为灰度图进行SVD处理
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # 应用自适应SVD
            denoised_gray, k = adaptive_svd(gray)

            # 将降噪后的灰度图转回RGB
            image = cv2.cvtColor(denoised_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            k = 0

        # 高斯模糊
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # 调整大小
        image = cv2.resize(image, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # 归一化
        return image.astype('float32') / 255.0, k  # 返回图像和k值

    except Exception as e:
        print(f"Preprocessing error: {e}")
        # 错误处理 - 使用灰度图像作为替代
        image = cv2.imread(image_path)
        if image is None:
            image = np.ones((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), dtype=np.float32) * 0.5
        else:
            image = cv2.resize(image, config.IMAGE_SIZE)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.astype('float32') / 255.0, 0


# Visualization function for SURF-2025-0145 effect
def visualize_svd_effect(image_path, output_dir):
    """可视化自适应SVD降噪效果"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 应用对比度增强
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 应用高斯模糊
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # 应用自适应SVD降噪
        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        denoised_gray, k = adaptive_svd(gray)  # 直接获取k值
        svd_image = cv2.cvtColor(denoised_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # 调整显示大小
        display_size = (384, 384)
        orig_display = cv2.resize(image, display_size)
        blur_display = cv2.resize(blurred, display_size)
        svd_display = cv2.resize(svd_image, display_size)

        # 创建对比图
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
        plt.title(f'Adaptive SURF-2025-0145 (k={k}, threshold={config.SVD_CHANGE_THRESHOLD})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'adaptive_svd_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(
            f"Adaptive SURF-2025-0145 comparison visualization saved to {os.path.join(output_dir, 'adaptive_svd_comparison.png')}")

    except Exception as e:
        print(f"Error in SURF-2025-0145 visualization: {e}")


# Mammography dataset class
class MammographyDataset(Dataset):
    def __init__(self, dataframe, directory, x_col, y_col=None, transform=None):
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.transform = transform
        self.k_values = []  # 存储每张图像的k值


        if y_col:
            # 获取唯一类别
            self.classes = sorted(self.dataframe[y_col].unique())
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
        image, k = enhanced_preprocessing(image_path)  # 获取图像和k值
        self.k_values.append(k)  # 记录k值

        if self.transform:
            image = self.transform(image)

        if self.y_col:
            label = self.class_to_idx[row[self.y_col]]
            return image, label
        else:
            return image, row[self.x_col]

    def get_k_stats(self):
        """返回k值统计信息"""
        if self.k_values:
            return {
                'min_k': min(self.k_values),
                'max_k': max(self.k_values),
                'mean_k': np.mean(self.k_values),
                'median_k': np.median(self.k_values)
            }
        return {}


# Data loading and analysis
def load_and_analyze_data(data_dir):
    """Load and analyze dataset"""
    print("Loading data...")
    train_path = r"D:\desktop\w2\w2\mammography_images\Training_set.csv"  # should be adjusted manually
    test_path = r"D:\desktop\w2\w2\mammography_images\Testing_set.csv"  # should be adjusted manually

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

# Data loader creation
def create_data_loaders(train_df, test_df, class_weights, data_dir):
    print("Creating data loaders...")

    # 训练数据增强
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 验证/测试数据增强
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 分割训练集和验证集
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['label']
    )
    print(f"Dataset split completed, training set: {len(train_df)}, validation set: {len(val_df)}")

    # 创建数据集
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

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2 if device.type != 'cuda' else 0,
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

    # 打印k值统计信息
    print("\nSURF-2025-0145 k-value statistics:")
    print(f"Training set: {train_dataset.get_k_stats()}")
    print(f"Validation set: {val_dataset.get_k_stats()}")
    print(f"Test set: {test_dataset.get_k_stats()}")

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


# Model building
def build_advanced_model(num_classes):
    print("Building model...")
    from torchvision.models import EfficientNet_B4_Weights

    # 加载预训练模型
    base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    features = base_model.features
    print("Pre-trained model loaded")

    # 解冻更多层
    for param in features.parameters():
        param.requires_grad = False
    for param in features[-10:].parameters():
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

    # 构建分类器
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


# Training function
def train_model(model, train_loader, val_loader, class_weights, unique_classes):
    print("Please enter the epoch patience:")
    n_patience = int(input())

    print("Starting model training...")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # 学习率调度器
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=n_patience, min_lr=config.MIN_LR)

    # 早停机制
    best_val_auc = 0.0
    patience = config.PATIENCE
    counter = 0
    best_epoch = 0

    # 训练历史记录
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
        print(f"\n===== Epoch {epoch + 1}/{config.EPOCHS} =====")
        log_resources()

        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_all_preds = []
        train_all_labels = []

        print("Training...")
        for i, (inputs, labels) in enumerate(train_loader):
            # 内存管理
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=config.USE_MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / config.ACCUMULATION_STEPS

            loss.backward()

            # 梯度累积更新
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * config.ACCUMULATION_STEPS * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0) * config.ACCUMULATION_STEPS
            correct += predicted.eq(labels).sum().item() * config.ACCUMULATION_STEPS

            # 收集预测结果
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())

            # 打印进度
            if (i + 1) % 20 == 0:
                print(f"Batch {i + 1}/{len(train_loader)}, loss: {loss.item() * config.ACCUMULATION_STEPS:.4f}")

            # 定期清理缓存
            if (i + 1) % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

        # 计算训练指标
        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_accuracy = 100. * correct / total
        train_micro_precision = precision_score(train_all_labels, train_all_preds, average='micro')
        train_micro_recall = recall_score(train_all_labels, train_all_preds, average='micro')

        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['train_precision'].append(train_micro_precision)
        history['train_recall'].append(train_micro_recall)

        print(f"Train loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.2f}%")
        print(f"Train precision: {train_micro_precision:.4f}, recall: {train_micro_recall:.4f}")

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
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

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
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = 100. * correct / total
        val_micro_precision = precision_score(all_labels, all_preds, average='micro')
        val_micro_recall = recall_score(all_labels, all_preds, average='micro')

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['val_precision'].append(val_micro_precision)
        history['val_recall'].append(val_micro_recall)

        print(f"Val loss: {epoch_val_loss:.4f}, accuracy: {epoch_val_accuracy:.2f}%")
        print(f"Val precision: {val_micro_precision:.4f}, recall: {val_micro_recall:.4f}")

        # 计算AUC
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
        print(f"Epoch time: {epoch_time:.2f}s, avg: {avg_epoch_time:.2f}s, remaining: {remaining_time / 60:.1f}min")

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
                print(f"Early stopping triggered, best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
                break

        # 清理缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    total_time = time.time() - start_time
    print(f"Training completed, total time: {total_time / 60:.2f} minutes")

    # 保存训练时间信息
    time_info = {
        'total_time': total_time,
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'epoch_times': epoch_times
    }
    with open(os.path.join(config.OUTPUT_DIR, 'training_time.json'), 'w') as f:
        json.dump(time_info, f)

    return history


# Visualization of results
def visualize_results(history, output_dir):
    """可视化训练指标"""
    plt.figure(figsize=(18, 12))
    os.makedirs(output_dir, exist_ok=True)

    # 1. 准确率曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
    plt.title('Accuracy Over Epochs')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. 损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. AUC曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['val_auc'], label='Validation AUC', marker='s')
    plt.title('AUC Over Epochs')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. 精确率和召回率
    plt.subplot(2, 2, 4)
    plt.plot(history['train_precision'], label='Training Precision', marker='o')
    plt.plot(history['val_precision'], label='Validation Precision', marker='s')
    plt.plot(history['train_recall'], label='Training Recall', marker='^')
    plt.plot(history['val_recall'], label='Validation Recall', marker='v')
    plt.title('Precision and Recall Over Epochs')
    plt.ylabel('Metric Value')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training metrics visualization saved to: {plot_path}")


# Model evaluation
def evaluate_model(model, test_loader, class_to_idx, output_dir, unique_classes, device):
    print("Starting model evaluation...")
    model.eval()
    results = []

    with torch.no_grad():
        for inputs, filenames in test_loader:
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

    # 保存结果
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation completed, results saved to {results_path}")
    print("Test set has no ground truth labels, classification report cannot be generated")
    return results


# Main function
def main():
    parser = argparse.ArgumentParser(description='Mammography Image Classification with Adaptive SURF-2025-0145')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='Data directory')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--use_svd', action='store_true', help='Enable SURF-2025-0145 denoising')
    parser.add_argument('--svd_threshold', type=float, default=0.1, help='SURF-2025-0145 change threshold')
    args = parser.parse_args()

    try:
        # 更新配置
        config.DATA_DIR = args.data_dir
        config.OUTPUT_DIR = args.output_dir
        config.USE_SVD = args.use_svd
        config.SVD_CHANGE_THRESHOLD = args.svd_threshold

        # 加载数据
        train_df, test_df, class_weights, unique_classes = load_and_analyze_data(config.DATA_DIR)

        # 创建数据加载器
        train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
            train_df, test_df, class_weights, config.DATA_DIR
        )

        # 构建模型
        model = build_advanced_model(len(unique_classes))

        # 训练模型
        history = train_model(model, train_loader, val_loader, class_weights, unique_classes)

        # 可视化结果
        visualize_results(history, config.OUTPUT_DIR)

        # 加载最佳模型进行评估
        best_model = build_advanced_model(len(unique_classes))
        best_model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth')))
        best_model.to(device).eval()

        # 评估模型
        evaluate_model(best_model, test_loader, class_to_idx, config.OUTPUT_DIR, unique_classes, device)

        # 保存最终模型
        final_model_path = os.path.join(config.OUTPUT_DIR, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Model saved as {final_model_path}")

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()