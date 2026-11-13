import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import cv2
import os
import gc
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import warnings
from tqdm import tqdm
import sys
import atexit


# 环境稳定性增强 - 启动时检查
# def check_environment():
#     """检查Python环境关键文件是否完整"""
#     required_files = [
#         "Lib/encodings/utf_8_sig.py",
#         "Lib/site.py",
#         "Lib/codecs.py"
#     ]
#
#     # 使用基础Python路径而非虚拟环境路径
#     base_path = sys.base_prefix if hasattr(sys, 'base_prefix') else sys.prefix
#     lib_path = os.path.join(base_path, "Lib")
#
#     for file in required_files:
#         # 使用正确的路径连接方式
#         path = os.path.join(lib_path, file)
#         # 规范化路径格式
#         path = os.path.normpath(path)
#
#         if not os.path.exists(path):
#             raise EnvironmentError(f"关键文件缺失: {path}")
#
#         # 检查文件大小是否正常
#         if os.path.getsize(path) < 1000:  # 正常应大于1KB
#             raise EnvironmentError(f"文件可能损坏: {path} (大小: {os.path.getsize(path)}字节)")
#
# print("Python环境检查通过")
#
# # 注册环境检查
# if not hasattr(sys, 'frozen'):
#     check_environment()


# 注册退出清理函数
@atexit.register
def cleanup_resources():
    """程序退出时强制清理资源"""
    print("执行资源清理...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


warnings.filterwarnings('ignore')

# 定义更强大的对比度增强核
CONTRAST_KERNEL = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 2, 2, 2, -1],
    [-1, 2, 8, 2, -1],
    [-1, 2, 2, 2, -1],
    [-1, -1, -1, -1, -1]
]) / 8.0

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检测GPU并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if device.type == "cuda":
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")

    # GPU内存保护 - 限制最大使用量
    torch.cuda.set_per_process_memory_fraction(0.8)  # 只使用80%的GPU内存
    # 配置内存分配器避免碎片化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# 超参数
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16  # 减小batch size防止内存溢出
EPOCHS = 50
LEARNING_RATE = 2e-5
BASE_MODEL = 'efficientnet_b4'
GRAD_ACCUM_STEPS = 2  # 梯度累积步数


# 资源监控
def log_resources():
    """记录系统资源使用情况"""
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


def load_and_analyze_data():
    """加载并分析数据集，增强数据验证和平衡处理"""
    train_path = r"D:\desktop\w2\w2\mammography_images\Training_set.csv"
    test_path = r"D:\desktop\w2\w2\mammography_images\Testing_set.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("训练或测试CSV文件未找到")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 数据完整性验证
    required_columns = {'filename', 'label'}
    if not required_columns.issubset(train.columns):
        raise ValueError("训练数据缺少必要列")
    if 'filename' not in test.columns:
        raise ValueError("测试数据缺少文件名列")

    # 高级类别分析
    class_counts = train['label'].value_counts()
    print(f"类别分布:\n{class_counts}")

    # 计算类别权重
    unique_classes = np.unique(train['label'])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train['label']
    )
    class_weights = dict(zip(unique_classes, class_weights))

    # 对少数类别应用更强的权重
    max_weight = max(class_weights.values())
    for cls in class_weights:
        if class_weights[cls] == max_weight:
            class_weights[cls] *= 1.5

    return train, test, class_weights


def enhanced_preprocessing(image, kernel=None, use_svd=False):
    """更先进的图像预处理流程"""
    try:
        # 转换为灰度图(如果是彩色)
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 数据标准化和类型转换
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 自适应直方图均衡化(CLAHE) - 提高对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # 高斯模糊去噪
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # 核增强(使用更强大的核)
        if kernel is not None:
            image = cv2.filter2D(image, -1, kernel)

        # 可选SVD处理(更稳定的实现)
        if use_svd:
            try:
                image_float = image.astype(np.float32) / 255.0
                U, s, Vt = np.linalg.svd(image_float, full_matrices=False)
                # 保留前30个奇异值(更激进的特征保留)
                s[30:] = 0
                image = np.dot(U, np.dot(np.diag(s), Vt))
                image = (image * 255).clip(0, 255).astype(np.uint8)
            except np.linalg.LinAlgError:
                pass

        # 调整大小(使用更高质量的插值)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LANCZOS4)

        # 转换为3通道(兼容预训练模型)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 标准化到[0,1]范围
        return image.astype('float32') / 255.0

    except Exception as e:
        print(f"预处理错误: {e}")
        # 后备处理方案
        image = cv2.resize(image, IMAGE_SIZE)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.astype('float32') / 255.0


class MammoDataset(Dataset):
    """自定义数据集类（增强稳定性）"""

    def __init__(self, dataframe, directory, transform=None, preprocess=True, is_test=False):
        self.dataframe = dataframe
        self.directory = directory
        self.transform = transform
        self.preprocess = preprocess
        self.is_test = is_test  # 区分训练/测试集

        # 仅训练集需要处理标签
        if not self.is_test:
            if 'label' not in dataframe.columns:
                raise ValueError("训练集必须包含'label'列")

            self.classes = sorted(dataframe['label'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """增强的错误处理和资源管理"""
        try:
            row = self.dataframe.iloc[idx]
            filename = row['filename']
            img_path = os.path.join(self.directory, filename)

            # 加载图像
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"无法加载图像: {img_path}")

            # 修复颜色空间
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 应用预处理
            if self.preprocess:
                image = enhanced_preprocessing(image, kernel=CONTRAST_KERNEL, use_svd=True)

            # 应用变换并确保格式为CHW
            if self.transform:
                image = self.transform(image)
            else:
                # 手动转换为PyTorch格式 [C, H, W]
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).permute(2, 0, 1).float()

            # 测试集只返回图像和文件名
            if self.is_test:
                return image, filename

            # 训练/验证集返回图像和标签
            label = row['label']
            label_idx = self.class_to_idx[label]
            return image, label_idx

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            # 返回空图像作为后备
            empty_img = torch.zeros((3, *IMAGE_SIZE), dtype=torch.float32)
            if self.is_test:
                return empty_img, filename
            return empty_img, 0  # 默认类别


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """空间注意力机制"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_out)
        return self.sigmoid(x_out) * x


def build_advanced_model(num_classes):
    """构建PyTorch模型（添加内存保护）"""
    # 加载预训练模型
    if BASE_MODEL == 'efficientnet_b4':
        model = models.efficientnet_b4(pretrained=True)
        in_features = model.classifier[1].in_features
    else:
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features

    # 冻结预训练层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后几层
    layers_to_unfreeze = 5
    for child in list(model.children())[-layers_to_unfreeze:]:
        for param in child.parameters():
            param.requires_grad = True

    # 添加注意力机制
    model.attention = nn.Sequential(
        ChannelAttention(in_features),
        SpatialAttention()
    )

    # 修改分类头
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.SiLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.6),
        nn.Linear(1024, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes))

    return model.to(device)


def train_model(model, train_loader, val_loader, class_weights):
    """训练PyTorch模型（增强资源管理）"""
    # 将类别权重转换为Tensor
    weights = torch.tensor([class_weights[cls] for cls in sorted(class_weights.keys())],
                           dtype=torch.float).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    best_roc_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_roc_auc': []}

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        # 资源监控
        print(f"\nEpoch {epoch + 1}/{EPOCHS} 资源状态:")
        log_resources()

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")):
            images, labels = images.to(device), labels.to(device)

            # 混合精度训练
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度累积
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # 清理缓存
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            train_loss += loss.item() * images.size(0) * GRAD_ACCUM_STEPS

        # 计算平均训练损失
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                # 获取预测结果
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # 定期清理内存
                if len(all_labels) % 100 == 0:
                    torch.cuda.empty_cache()

        # 计算验证指标
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_roc_auc'].append(val_roc_auc)

        # 打印进度
        print(f"Epoch {epoch + 1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Acc: {val_acc:.4f} - "
              f"Val ROC AUC: {val_roc_auc:.4f}")

        # 更新学习率
        scheduler.step(val_roc_auc)

        # 保存最佳模型
        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            torch.save(model.state_dict(), 'best_model.pth')
            print("保存新的最佳模型")

        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

    return history


def create_data_loaders(train_df, test_df, class_weights):
    """创建PyTorch数据加载器（Windows优化）"""
    # 定义数据增强
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.25, 0.25), shear=0.2, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df['label'], random_state=42
    )

    # 创建数据集
    train_dataset = MammoDataset(
        train_df,
        directory=r"D:\desktop\w2\w2\mammography_images\train",
        transform=train_transform
    )

    val_dataset = MammoDataset(
        val_df,
        directory=r"D:\desktop\w2\w2\mammography_images\train",
        transform=val_transform
    )

    test_dataset = MammoDataset(
        test_df,
        directory=r"D:\desktop\w2\w2\mammography_images\test",
        transform=val_transform,
        preprocess=True,
        is_test=True  # 标记为测试集
    )

    # 创建数据加载器 (Windows兼容性优化)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows下禁用多进程
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Windows下禁用多进程
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Windows下禁用多进程
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


def evaluate_model(model, test_loader, class_to_idx):
    """评估模型（修复测试集处理）"""
    model.eval()
    all_preds = []
    all_filenames = []

    with torch.no_grad():
        for images, filenames in test_loader:  # 测试集返回 (image, filename)
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_filenames.extend(filenames)

    # 反转标签映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred_labels = [idx_to_class[pred] for pred in all_preds]

    # 保存结果
    results = pd.DataFrame({
        'filename': all_filenames,
        'prediction': pred_labels
    })
    results.to_csv('predictions.csv', index=False)

    return results


def visualize_results(history):
    """可视化训练结果"""
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # AUC曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['val_roc_auc'], label='Val ROC AUC')
    plt.title('ROC AUC')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    plt.close()


def main():
    try:
        # 加载和分析数据
        train_df, test_df, class_weights = load_and_analyze_data()

        # 创建数据加载器
        train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(train_df, test_df, class_weights)

        # 构建模型
        model = build_advanced_model(len(class_to_idx))

        # 训练模型
        print("开始训练模型...")
        history = train_model(model, train_loader, val_loader, class_weights)

        # 训练结果可视化
        visualize_results(history)

        # 评估模型
        print("开始模型评估...")
        try:
            results = evaluate_model(model, test_loader, class_to_idx)
            print("评估完成! 结果已保存为 predictions.csv")
        except Exception as e:
            print(f"评估过程中出现错误: {e}")

        # 保存最终模型
        torch.save(model.state_dict(), 'final_advanced_model.pth')
        print("模型已保存为 final_advanced_model.pth")

    except Exception as e:
        print(f"主程序发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源清理
        cleanup_resources()


if __name__ == '__main__':
    # 添加额外的错误处理层
    try:
        main()
    except Exception as e:
        print(f"程序崩溃: {str(e)}")
        # 尝试优雅退出
        cleanup_resources()
        sys.exit(1)