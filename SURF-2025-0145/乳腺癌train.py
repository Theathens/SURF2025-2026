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

# 设置随机种子以保证可复现性
np.random.seed(42)
torch.manual_seed(42)

# 检测并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    # GPU内存保护 - 限制最大使用量
    torch.cuda.set_per_process_memory_fraction(0.8)  # 仅使用80%的内存
    # 配置内存分配器避免碎片化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# 资源监控函数
def log_resources():
    if device.type == "cuda":
        alloc_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU内存使用: {alloc_mem:.2f}/{total_mem:.2f} GB ({alloc_mem / total_mem:.1%})")
    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        print(f"CPU使用率: {cpu_percent}% | 内存使用率: {ram_percent}%")
    except ImportError:
        pass


# 超参数配置
print("请输入训练轮数:")
num = int(input())
print("请输入k值（推荐50）")
k_value = int(input())


class Config:
    IMAGE_SIZE = (384, 384)  # 调整为384x384以平衡精度和内存
    BATCH_SIZE = 16  # 批次大小
    ACCUMULATION_STEPS = 2  # 梯度累积步数，等效于BATCH_SIZE=32
    EPOCHS = num
    LEARNING_RATE = 5e-5  # 降低学习率
    WEIGHT_DECAY = 1e-5
    BASE_MODEL = 'efficientnet_b4'
    PATIENCE = 10
    MIN_LR = 1e-7
    DATA_DIR = r"D:\desktop\w3\w3\mammography_images"  # 训练和测试图像的输入路径，需手动调整
    OUTPUT_DIR = r"C:\Users\27785\PycharmProjects\PythonProject1\.venv\乳腺癌w4\output(2)"  # 结果输出路径，需手动调整
    USE_MIXED_PRECISION = True
    USE_SVD = True  # 新增：是否使用SVD降噪
    SVD_COMPONENTS = k_value  # 新增：SVD保留的主成分数量


config = Config()


# 基于SVD的图像降噪函数
def apply_svd(image, k):
    """
    应用SVD对图像进行降噪和压缩，增强数值稳定性
    image: numpy数组（uint8），范围[0, 255]
    返回: numpy数组（float32），范围[0, 255]
    """
    k = k_value
    print(f"SVD压缩，k = {Config.SVD_COMPONENTS}...")
    try:
        # 转换为float32
        gray = image.astype(np.float32)
        # 将图像归一化到[0, 1]
        min_val = np.min(gray)
        max_val = np.max(gray)
        # 避免除零
        if max_val == min_val:
            return gray
        normalized = (gray - min_val) / (max_val - min_val + 1e-8)
        # 执行SVD
        U, S, Vt = np.linalg.svd(normalized, full_matrices=False)
        # 截断到前k个奇异值
        k = min(k, len(S))
        S_trunc = np.diag(S[:k])
        reconstructed = U[:, :k] @ S_trunc @ Vt[:k, :]
        # 反归一化
        reconstructed = reconstructed * (max_val - min_val) + min_val
        reconstructed = np.clip(reconstructed, 0, 255)
        return reconstructed.astype(np.float32)
    except Exception as e:
        print(f"SVD错误: {e}")
        return image.astype(np.float32)


def enhanced_preprocessing(image_path):
    """增强的图像预处理管道，包含SVD降噪选项"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        # 确保图像为RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 增强对比度
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # 如果启用则应用SVD降噪
        if config.USE_SVD:
            # 转换为灰度图以进行SVD
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # 应用SVD
            denoised_gray = apply_svd(gray, config.SVD_COMPONENTS)
            # 转换回RGB（通过堆叠相同通道）
            image = cv2.cvtColor(denoised_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # 高斯模糊
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # 调整大小为384x384
        image = cv2.resize(image, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        # 归一化
        return image.astype('float32') / 255.0
    except Exception as e:
        print(f"预处理错误: {e}")
        # 错误恢复 - 使用灰度图替代
        image = cv2.imread(image_path)
        if image is None:
            image = np.ones((config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3), dtype=np.float32) * 0.5
        else:
            image = cv2.resize(image, config.IMAGE_SIZE)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.astype('float32') / 255.0


def visualize_svd_effect(image_path, output_dir):
    """可视化样本图像的SVD降噪效果"""
    try:
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        # 读取并预处理（不使用SVD）
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
        # 应用SVD降噪
        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        denoised_gray = apply_svd(gray, config.SVD_COMPONENTS)
        svd_image = cv2.cvtColor(denoised_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # 调整大小以显示
        display_size = (384, 384)
        orig_display = cv2.resize(image, display_size)
        blur_display = cv2.resize(blurred, display_size)
        svd_display = cv2.resize(svd_image, display_size)
        # 创建对比图
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(orig_display)
        plt.title('原始图像')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(blur_display)
        plt.title('模糊图像')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title(f'SVD降噪 (k={k_value})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'svd_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SVD对比可视化已保存至 {os.path.join(output_dir, 'svd_comparison.png')}")
    except Exception as e:
        print(f"SVD可视化错误: {e}")


class MammographyDataset(Dataset):
    """乳腺X光图像数据集类"""

    def __init__(self, dataframe, directory, x_col, y_col=None, transform=None):
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.transform = transform
        if y_col:
            self.classes = sorted(dataframe[y_col].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            print(f"数据集类别映射: {self.class_to_idx}")
        else:
            self.classes = []
            self.class_to_idx = {}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.directory, row[self.x_col])
        image = enhanced_preprocessing(image_path)
        if self.transform:
            image = self.transform(image)
        if self.y_col:
            label = self.class_to_idx[row[self.y_col]]
            return image, label
        else:
            return image, row[self.x_col]


def load_and_analyze_data(data_dir):
    """加载并分析数据集"""
    print("加载数据...")
    train_path = r"D:\desktop\w2\w2\mammography_images\Training_set.csv"  # 需手动调整
    test_path = r"D:\desktop\w2\w2\mammography_images\Testing_set.csv"  # 需手动调整
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"未找到训练或测试CSV文件: {train_path}, {test_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"数据加载完成，训练集大小: {len(train)}, 测试集大小: {len(test)}")
    # 数据完整性验证
    required_columns = {'filename', 'label'}
    assert required_columns.issubset(train.columns), "训练数据缺少必要列"
    assert 'filename' in test.columns, "测试数据缺少filename列"
    # 类别分布分析
    class_counts = train['label'].value_counts()
    print(f"类别分布:\n{class_counts}")
    # 计算类别权重
    unique_classes = np.unique(train['label'])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train['label']
    )
    # 对少数类应用更强的权重
    max_weight = max(class_weights)
    for i, weight in enumerate(class_weights):
        if weight < max_weight * 0.2:  # 增强代表性特别不足的类别的权重
            class_weights[i] *= 3.0
        elif weight < max_weight * 0.5:
            class_weights[i] *= 2.0
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("计算的类别权重:", class_weights)
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # 可视化样本图像的SVD效果
    sample_image = os.path.join(data_dir, 'train', train.iloc[0]['filename'])
    visualize_svd_effect(sample_image, config.OUTPUT_DIR)
    return train, test, class_weights, unique_classes


def create_data_loaders(train_df, test_df, class_weights, data_dir):
    """创建数据加载器"""
    print("创建数据加载器...")
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
    # 将训练集拆分为训练和验证集 - 确保分层抽样
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['label'] if 'label' in train_df.columns else None
    )
    print(f"数据集拆分完成，训练集: {len(train_df)}, 验证集: {len(val_df)}")
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
        num_workers=2 if device.type != 'cuda' else 0,  # CUDA设备使用0个工作进程
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
    print("数据加载器创建完成")
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def build_advanced_model(num_classes):
    """构建增强的EfficientNet-B4模型"""
    print("构建模型...")
    from torchvision.models import EfficientNet_B4_Weights
    # 加载预训练的EfficientNet-B4模型
    base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    features = base_model.features
    print("预训练模型加载完成")
    # 解冻更多层
    for param in features.parameters():
        param.requires_grad = False
    for param in features[-10:].parameters():  # 解冻更多层
        param.requires_grad = True
    print("模型参数已解冻")
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
    print(f"实际特征维度: {features_dim}")
    # 构建分类头
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
    print(f"模型总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
    return full_model


def train_model(model, train_loader, val_loader, class_weights, unique_classes):
    """训练模型"""
    print("请输入轮次耐心值:")
    n = int(input())
    print("开始模型训练...")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # 学习率调度器
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=n,
                                                      min_lr=config.MIN_LR)
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
        print(f"\n===== 轮次 {epoch + 1}/{config.EPOCHS} =====")
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_all_preds = []
        train_all_labels = []
        train_all_probs = []  # 新增：存储训练集预测概率
        print("训练中...")
        for i, (inputs, labels) in enumerate(train_loader):
            # 管理内存使用
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=config.USE_MIXED_PRECISION):
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
            # 收集训练预测结果用于计算精确率和召回率
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())
            # 新增：收集训练集预测概率（用于AUC计算）
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            train_all_probs.extend(probs)
            # 打印进度
            if (i + 1) % 20 == 0:
                print(
                    f"处理批次 {i + 1}/{len(train_loader)}, 当前批次损失: {loss.item() * config.ACCUMULATION_STEPS:.4f}")
            # 定期清理缓存
            if (i + 1) % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        # 计算训练集精确率和召回率
        train_micro_precision = precision_score(train_all_labels, train_all_preds, average='micro')
        train_micro_recall = recall_score(train_all_labels, train_all_preds, average='micro')

        # 新增：计算训练集AUC
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
        history['train_auc'].append(train_auc)  # 记录训练集AUC

        print(f"训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_accuracy:.2f}%")
        print(f"训练精确率: {train_micro_precision:.4f}, 训练召回率: {train_micro_recall:.4f}")
        print(f"训练AUC: {train_auc:.4f}")  # 打印训练集AUC

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        print("验证中...")
        with torch.no_grad():
            for inputs, labels in val_loader:
                # 管理内存使用
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

        # 计算验证集精确率和召回率
        val_micro_precision = precision_score(all_labels, all_preds, average='micro')
        val_micro_recall = recall_score(all_labels, all_preds, average='micro')
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = 100. * correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        history['val_precision'].append(val_micro_precision)
        history['val_recall'].append(val_micro_recall)
        print(f"验证损失: {epoch_val_loss:.4f}, 验证准确率: {epoch_val_accuracy:.2f}%")
        print(f"验证精确率: {val_micro_precision:.4f}, 验证召回率: {val_micro_recall:.4f}")

        # 计算AUC
        num_classes = len(unique_classes)
        if num_classes == 2:
            val_auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
            history['val_auc'].append(val_auc)
            print(f"验证AUC: {val_auc:.4f}")
        else:
            from sklearn.preprocessing import label_binarize
            all_labels_onehot = label_binarize(all_labels, classes=list(range(num_classes)))
            val_auc = roc_auc_score(all_labels_onehot, all_probs, average='macro', multi_class='ovr')
            history['val_auc'].append(val_auc)
            print(f"验证宏平均AUC: {val_auc:.4f}")

        # 更新学习率
        scheduler1.step()
        scheduler2.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr}")

        # 计算轮次时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = (config.EPOCHS - epoch - 1) * avg_epoch_time
        print(f"轮次时间: {epoch_time:.2f}s, 平均: {avg_epoch_time:.2f}s, 剩余: {remaining_time / 60:.1f}分钟")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            counter = 0
            model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"最佳模型已保存至 {model_path}, 验证AUC: {best_val_auc:.4f}")
        else:
            counter += 1
            print(f"早停计数器: {counter}/{patience}")
            if counter >= patience:
                print(f"触发早停, 最佳验证AUC: {best_val_auc:.4f} (在轮次 {best_epoch})")
                break

        # 每轮次后清理缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    total_time = time.time() - start_time
    print(f"训练完成, 总时间: {total_time / 60:.2f} 分钟")
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
    """评估模型并生成详细报告"""
    print("开始模型评估...")
    model.eval()
    results = []
    all_preds = []
    all_labels = []  # 注意: 测试集可能没有标签，这里存储文件名
    all_filenames = []
    all_probs = []
    with torch.no_grad():
        for inputs, filenames in test_loader:
            # 管理内存使用
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
            all_filenames.extend(filenames)  # 存储文件名而非标签
            all_probs.extend(probs)
            # 定期清理缓存
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 保存结果
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"评估完成, 结果已保存至 {results_path}")
    # 测试集通常没有真实标签，因此无法生成分类报告
    print("测试集没有真实标签, 无法生成分类报告")
    return results


def visualize_results(history, output_dir):
    """增强的训练指标可视化"""
    plt.figure(figsize=(18, 12))
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 准确率曲线
    plt.subplot(2, 2, 1)
    if 'train_accuracy' in history and 'val_accuracy' in history:
        plt.plot(history['train_accuracy'], label='训练准确率', marker='o')
        plt.plot(history['val_accuracy'], label='验证准确率', marker='s')
        plt.title('准确率随轮次变化 (使用SVD)')
        plt.ylabel('准确率 (%)')
    else:
        plt.text(0.5, 0.5, '无准确率数据', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('准确率数据缺失')
    plt.xlabel('轮次')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. 损失曲线
    plt.subplot(2, 2, 2)
    if 'train_loss' in history and 'val_loss' in history:
        plt.plot(history['train_loss'], label='训练损失', marker='o')
        plt.plot(history['val_loss'], label='验证损失', marker='s')
        plt.title('损失随轮次变化 (使用SVD)')
        plt.ylabel('损失')
    else:
        plt.text(0.5, 0.5, '无损失数据', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('损失数据缺失 (使用SVD)')
    plt.xlabel('轮次')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. AUC曲线
    plt.subplot(2, 2, 3)
    if 'train_auc' in history and 'val_auc' in history:
        plt.plot(history['train_auc'], label='训练AUC', marker='o')
        plt.plot(history['val_auc'], label='验证AUC', marker='s')
        plt.title('AUC随轮次变化 (使用SVD)')
        plt.ylabel('AUC')
    elif 'train_macro_auc' in history and 'val_macro_auc' in history:
        plt.plot(history['train_macro_auc'], label='训练宏平均AUC', marker='o')
        plt.plot(history['val_macro_auc'], label='验证宏平均AUC', marker='s')
        plt.title('宏平均AUC随轮次变化 (使用SVD)')
        plt.ylabel('AUC')
    else:
        plt.text(0.5, 0.5, '无AUC数据', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('AUC数据缺失')
    plt.xlabel('轮次')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. 精确率和召回率曲线
    plt.subplot(2, 2, 4)
    if 'train_precision' in history and 'val_precision' in history and 'train_recall' in history and 'val_recall' in history:
        plt.plot(history['train_precision'], label='训练精确率', marker='o')
        plt.plot(history['val_precision'], label='验证精确率', marker='s')
        plt.plot(history['train_recall'], label='训练召回率', marker='^')
        plt.plot(history['val_recall'], label='验证召回率', marker='v')
        plt.title('精确率和召回率随轮次变化 (使用SVD)')
        plt.ylabel('指标值')
    else:
        plt.text(0.5, 0.5, '无精确率/召回率数据', ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.title('精确率/召回率数据缺失')
    plt.xlabel('轮次')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练指标可视化已保存至: {plot_path}")


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='乳腺X光图像分类模型')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, help='数据目录')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help='输出目录')
    parser.add_argument('--use_svd', action='store_true', help='启用SVD降噪')
    parser.add_argument('--svd_components', type=int, default=50, help='SVD保留的主成分数量')
    args = parser.parse_args()

    try:
        # 更新配置
        config.DATA_DIR = args.data_dir
        config.OUTPUT_DIR = args.output_dir
        config.USE_SVD = args.use_svd
        config.SVD_COMPONENTS = args.svd_components

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

        # 可视化结果
        visualize_results(history, config.OUTPUT_DIR)

        # 加载最佳模型进行评估
        best_model = build_advanced_model(len(class_to_idx))
        best_model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, 'best_model.pth')))
        best_model.to(device).eval()

        # 评估模型
        evaluate_model(best_model, test_loader, class_to_idx, config.OUTPUT_DIR, unique_classes, device)

        # 保存最终模型
        final_model_path = os.path.join(config.OUTPUT_DIR, 'final_advanced_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"模型已保存为 {final_model_path}")

    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()