"""
train_unet_effb0.py
====================
在 Train_unet_effb0.py 基础上，移植 unet_replace_SVD.py 的早停机制。

新增改动（相较上一版）
─────────────────────────────────────────────────
[早停] 参考 unet_replace_SVD.py 的 train_model() 完整移植：
  1. 变量初始化: best_val_loss / early_stop_counter / best_epoch
  2. 每轮耗时统计: epoch_times 列表，实时打印"本轮耗时 / 平均耗时 / 预计剩余"
  3. 早停判断逻辑:
       - Val Loss 下降 → 重置 counter，保存最优 checkpoint，记录 best_epoch
       - Val Loss 不降 → counter+1，打印 counter/patience
       - counter >= patience → 打印触发信息并 break
  4. 训练结束后将 total_time / avg_epoch_time / epoch_times
     写入 output_dir/training_time.json（与 SVD 脚本格式完全一致）
  5. 配置块新增 patience 参数（默认 10）

其余结构（EfficientNet-B0 Encoder、差异化学习率、
类别权重、CosineAnnealingWarmRestarts、断点续训）均保持不变。
─────────────────────────────────────────────────
"""

import os
# 设置 Hugging Face 镜像（仅首次下载预训练权重时需要）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import warnings
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import timm

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.8)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# ============================================================
# 工具函数
# ============================================================
def log_resources():
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU memory: {alloc:.2f}/{total:.2f} GB ({alloc/total:.1%})")
    try:
        import psutil
        print(f"CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%")
    except ImportError:
        pass


def _double_conv(in_ch, out_ch):
    """两层 3×3 Conv + BatchNorm + ReLU（Decoder 基本单元）"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ============================================================
# 模型: EfficientNet-B0 Encoder + UNet Decoder
# ============================================================
class EfficientUNet(nn.Module):
    """
    UNet，Encoder 替换为 EfficientNet-B0（timm）。

    EfficientNet-B0 在 512×512 输入下各级特征图尺寸与通道数：
        e0: (B, 16,  256, 256)   stride=2
        e1: (B, 24,  128, 128)   stride=4
        e2: (B, 40,  64,  64)    stride=8
        e3: (B, 112, 32,  32)    stride=16
        e4: (B, 320, 16,  16)    stride=32

    Decoder 做 4 次 ×2 上采样，每次 cat 对应 skip feature，
    最终恢复到 stride=2（256×256），再接一次 ×2 到原始分辨率（512×512）。

    参数:
        num_classes  : 输出类别数（含背景）
        pretrained   : True 时加载 ImageNet 预训练权重（强烈推荐）
        decoder_ch   : Decoder 各级通道数列表（从深到浅），默认 [256,128,64,32,16]
    """

    # EfficientNet-B0 各级输出通道（固定，由 timm 决定）
    ENC_CHANNELS = [16, 24, 40, 112, 320]   # e0..e4

    def __init__(self, num_classes=2, pretrained=True, decoder_ch=None):
        super().__init__()

        if decoder_ch is None:
            decoder_ch = [256, 128, 64, 32, 16]
        assert len(decoder_ch) == 5, "decoder_ch 需要 5 个值"

        # ── Encoder：EfficientNet-B0，取全部 5 级中间特征 ──────────
        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),   # e0 ~ e4
        )

        e = self.ENC_CHANNELS   # 简写

        # ── Bottleneck：对 e4(320ch) 再做一次双卷积压缩 ────────────
        self.bottleneck = _double_conv(e[4], decoder_ch[0])

        # ── Decoder（从深到浅，共 4 次上采样） ──────────────────────
        # dec4: bottleneck(256) + e3(112) → 128
        self.upconv4  = nn.ConvTranspose2d(decoder_ch[0], decoder_ch[0], kernel_size=2, stride=2)
        self.decoder4 = _double_conv(decoder_ch[0] + e[3], decoder_ch[1])

        # dec3: dec4(128) + e2(40) → 64
        self.upconv3  = nn.ConvTranspose2d(decoder_ch[1], decoder_ch[1], kernel_size=2, stride=2)
        self.decoder3 = _double_conv(decoder_ch[1] + e[2], decoder_ch[2])

        # dec2: dec3(64) + e1(24) → 32
        self.upconv2  = nn.ConvTranspose2d(decoder_ch[2], decoder_ch[2], kernel_size=2, stride=2)
        self.decoder2 = _double_conv(decoder_ch[2] + e[1], decoder_ch[3])

        # dec1: dec2(32) + e0(16) → 16
        self.upconv1  = nn.ConvTranspose2d(decoder_ch[3], decoder_ch[3], kernel_size=2, stride=2)
        self.decoder1 = _double_conv(decoder_ch[3] + e[0], decoder_ch[4])

        # dec0: 最后一次上采样回到原始分辨率（无 skip）
        self.upconv0  = nn.ConvTranspose2d(decoder_ch[4], decoder_ch[4], kernel_size=2, stride=2)
        self.decoder0 = _double_conv(decoder_ch[4], decoder_ch[4])

        # ── 输出头 ──────────────────────────────────────────────────
        self.head = nn.Conv2d(decoder_ch[4], num_classes, kernel_size=1)

    def forward(self, x):
        e0, e1, e2, e3, e4 = self.encoder(x)

        b  = self.bottleneck(e4)

        d4 = self.upconv4(b)
        d4 = self._pad_cat(d4, e3)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = self._pad_cat(d3, e2)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = self._pad_cat(d2, e1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = self._pad_cat(d1, e0)
        d1 = self.decoder1(d1)

        d0 = self.upconv0(d1)
        d0 = self.decoder0(d0)

        return self.head(d0)

    @staticmethod
    def _pad_cat(up, skip):
        """上采样结果与 skip feature 尺寸可能差 1px，pad 后再 cat。"""
        dh = skip.size(2) - up.size(2)
        dw = skip.size(3) - up.size(3)
        if dh != 0 or dw != 0:
            up = torch.nn.functional.pad(
                up, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
            )
        return torch.cat([up, skip], dim=1)

    def encoder_parameters(self):
        """返回 EfficientNet Encoder 的参数（用于差异化学习率）"""
        return self.encoder.parameters()

    def decoder_parameters(self):
        """返回 Decoder + Bottleneck + Head 的参数"""
        decoder_modules = [
            self.bottleneck,
            self.upconv4, self.decoder4,
            self.upconv3, self.decoder3,
            self.upconv2, self.decoder2,
            self.upconv1, self.decoder1,
            self.upconv0, self.decoder0,
            self.head,
        ]
        for m in decoder_modules:
            yield from m.parameters()


# ============================================================
# 数据集
# ============================================================
class SegmentationDataset(Dataset):
    """
    读取 (图像 patch, 掩码 patch) 对。
    掩码为单通道 PNG，像素值 = 类别索引（0=背景）。
    augment=True 时，图像与掩码做同步几何变换。
    """

    def __init__(self, image_paths, mask_paths, input_size=(512, 512), augment=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.input_size  = input_size
        self.augment     = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            image = np.ones((*self.input_size, 3), dtype=np.uint8) * 128
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(self.input_size, dtype=np.uint8)

        wh = (self.input_size[1], self.input_size[0])
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, wh, interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != self.input_size:
            mask  = cv2.resize(mask,  wh, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1, :])
                mask  = np.ascontiguousarray(mask[:,  ::-1])
            if np.random.rand() < 0.5:
                image = np.ascontiguousarray(image[::-1, :, :])
                mask  = np.ascontiguousarray(mask[ ::-1, :])
            k = np.random.randint(0, 4)
            if k:
                image = np.ascontiguousarray(np.rot90(image, k))
                mask  = np.ascontiguousarray(np.rot90(mask,  k))

        image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask  = torch.from_numpy(mask.astype(np.int64))
        return image, mask


# ============================================================
# 类别权重自动计算（缓解类别不平衡）
# ============================================================
def compute_class_weights(mask_paths, num_classes, max_samples=500):
    """
    随机抽取至多 max_samples 张掩码，统计各类像素频率，
    返回 inverse-frequency 权重（稀有类权重更大）。
    """
    print(f"正在统计类别权重（采样 {min(max_samples, len(mask_paths))} 张掩码）...")
    counts = np.zeros(num_classes, dtype=np.int64)
    sample = np.random.choice(mask_paths,
                              size=min(max_samples, len(mask_paths)),
                              replace=False)
    for p in sample:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        for c in range(num_classes):
            counts[c] += int((m == c).sum())

    counts  = np.maximum(counts, 1)
    freq    = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * num_classes   # 归一化到均值=1
    print(f"类别权重: {np.round(weights, 3).tolist()}")
    return weights.astype(np.float32)


# ============================================================
# 数据配对 & DataLoader
# ============================================================
def create_data_from_directory(image_dir, mask_dir, test_size=0.2):
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(supported))

    image_paths, mask_paths = [], []
    for f in files:
        mp = os.path.join(mask_dir, f)
        if os.path.exists(mp):
            image_paths.append(os.path.join(image_dir, f))
            mask_paths.append(mp)

    if not image_paths:
        raise ValueError(f"在 {image_dir}/{mask_dir} 中未找到匹配的图像-掩码对")

    print(f"共找到 {len(image_paths)} 对图像-掩码 patch")
    ti, vi, tm, vm = train_test_split(image_paths, mask_paths,
                                       test_size=test_size, random_state=42)
    print(f"训练集: {len(ti)}, 验证集: {len(vi)}")
    return (ti, tm), (vi, vm)


def create_data_loaders(train_data, val_data, input_size=(512, 512), batch_size=4):
    try:
        import psutil
        nw = min(4, psutil.cpu_count(logical=True))
    except ImportError:
        nw = 0

    train_ds = SegmentationDataset(train_data[0], train_data[1], input_size, augment=True)
    val_ds   = SegmentationDataset(val_data[0],   val_data[1],   input_size, augment=False)

    kw = dict(num_workers=nw, pin_memory=True,
              prefetch_factor=2 if nw > 0 else None,
              persistent_workers=nw > 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    print(f"DataLoader: batch={batch_size}, num_workers={nw}")
    return train_loader, val_loader


# ============================================================
# mIoU
# ============================================================
def compute_miou(preds, targets, num_classes):
    ious = []
    for c in range(num_classes):
        p = preds == c
        t = targets == c
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append((p & t).sum().item() / union)
    return sum(ious) / len(ious) if ious else 0.0


# ============================================================
# 训练（含早停机制，移植自 unet_replace_SVD.py）
# ============================================================
def train_unet(model, train_loader, val_loader, num_classes,
               epochs=50, learning_rate=5e-4,
               save_path="unet_effb0_best.pth",
               output_dir="./output",
               class_weights=None,
               resume=None,
               patience=10):
    """
    训练函数。

    早停机制（移植自 unet_replace_SVD.py train_model()）:
    ┌──────────────────────────────────────────────────────────────┐
    │  变量: best_val_loss / early_stop_counter / best_epoch       │
    │  每轮: 记录 epoch_time，打印"本轮/平均/预计剩余"              │
    │  判断: Val Loss 下降 → 重置 counter，保存 checkpoint         │
    │        Val Loss 不降 → counter+1，打印 counter/patience      │
    │        counter >= patience → 打印触发信息，break              │
    │  结束: 将耗时信息写入 output_dir/training_time.json           │
    └──────────────────────────────────────────────────────────────┘

    差异化学习率:
        Encoder (EfficientNet-B0) → lr × 0.1
        Decoder / Bottleneck / Head → lr

    学习率调度: CosineAnnealingWarmRestarts（T_0=10 轮一次重启）
    断点续训:   resume 传入 checkpoint 路径时，恢复 model+optimizer+epoch
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 损失函数 ─────────────────────────────────────────────────
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # ── 差异化学习率优化器 ────────────────────────────────────────
    optimizer = optim.AdamW([
        {"params": model.encoder_parameters(), "lr": learning_rate * 0.1},
        {"params": model.decoder_parameters(), "lr": learning_rate},
    ], weight_decay=1e-4)

    # ── CosineAnnealingWarmRestarts ───────────────────────────────
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    start_epoch = 0

    # ── 断点续训 ──────────────────────────────────────────────────
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"已从 {resume} 恢复，继续从第 {start_epoch + 1} 轮训练")

    history = {"train_loss": [], "val_loss": [], "val_miou": []}

    # ── 早停变量（移植自 unet_replace_SVD.py） ───────────────────
    best_val_loss       = float("inf")   # 监控指标：Val Loss（越低越好）
    early_stop_counter  = 0              # 连续未改善轮数
    best_epoch          = 0              # 最优 checkpoint 对应的轮次
    epoch_times         = []             # 每轮耗时列表，用于预估剩余时间

    print(f"开始训练 EfficientUNet 分割模型（早停 patience={patience}）...")
    total_start = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()   # ← 移植自 SVD：记录每轮开始时间

        # ── Train ────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for data, targets in pbar:
            data    = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(data)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step(epoch + 1)

        # ── Validation ───────────────────────────────────────────
        model.eval()
        val_loss = miou_sum = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for data, targets in vbar:
                data    = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(data)
                loss    = criterion(outputs, targets)
                val_loss  += loss.item()
                preds      = torch.argmax(outputs, dim=1)
                miou_sum  += compute_miou(preds, targets, num_classes)
                vbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        avg_miou     = miou_sum / len(val_loader)

        # ── 耗时统计与剩余时间预估（移植自 unet_replace_SVD.py） ──
        epoch_time     = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = (epochs - epoch - 1) * avg_epoch_time   # 单位：秒

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_miou"].append(avg_miou)

        tqdm.write(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val mIoU: {avg_miou:.4f} | "
            f"LR(enc): {optimizer.param_groups[0]['lr']:.2e} | "
            f"LR(dec): {optimizer.param_groups[1]['lr']:.2e}"
        )
        # 移植自 SVD：打印耗时与剩余时间
        tqdm.write(
            f"Epoch time: {epoch_time:.2f}s | "
            f"Average: {avg_epoch_time:.2f}s | "
            f"Remaining: {remaining_time / 60:.1f} minutes"
        )

        # ── 早停判断（移植自 unet_replace_SVD.py，指标改为 Val Loss） ──
        if avg_val_loss < best_val_loss:
            # Val Loss 下降 → 改善，重置计数器，保存 checkpoint
            best_val_loss      = avg_val_loss
            best_epoch         = epoch + 1
            early_stop_counter = 0
            torch.save({
                "epoch":     epoch + 1,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  avg_val_loss,
                "val_miou":  avg_miou,
            }, save_path)
            tqdm.write(
                f"  → 已保存最优模型至 {save_path}（Val Loss: {best_val_loss:.6f}，Epoch {best_epoch}）"
            )
        else:
            # Val Loss 未下降 → 计数器 +1
            early_stop_counter += 1
            tqdm.write(
                f"  Early stopping counter: {early_stop_counter}/{patience}"
            )
            if early_stop_counter >= patience:
                tqdm.write(
                    f"  Early stopping triggered! "
                    f"Best Val Loss: {best_val_loss:.6f} at Epoch {best_epoch}"
                )
                break   # ← 提前结束训练循环

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── 训练结束：汇总耗时并写入 JSON（移植自 unet_replace_SVD.py） ──
    total_time = time.time() - total_start
    print(f"\n训练总耗时: {total_time:.1f} s ({total_time / 60:.1f} min)")
    print(f"最优模型出现在第 {best_epoch} 轮，Val Loss: {best_val_loss:.6f}")

    time_info = {
        "total_time":     total_time,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times) if epoch_times else 0,
        "epoch_times":    epoch_times,
    }
    time_json_path = os.path.join(output_dir, "training_time.json")
    with open(time_json_path, "w") as f:
        json.dump(time_info, f, indent=2)
    print(f"耗时信息已写入: {time_json_path}")

    return history


# ============================================================
# 可视化
# ============================================================
def visualize_results(history, output_dir):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss",   marker="o")
    plt.plot(history["val_loss"],   label="Validation Loss", marker="s")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(history["val_miou"], label="Validation mIoU", color="green", marker="s")
    plt.title("Validation mIoU")
    plt.xlabel("Epoch"); plt.ylabel("mIoU")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    path = os.path.join(output_dir, "unet_effb0_training_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存至: {path}")


# ============================================================
# 推理
# ============================================================
PALETTE = np.array([
    [0,   0,   0  ], [255, 0,   0  ], [0,   255, 0  ], [0,   0,   255],
    [255, 255, 0  ], [255, 0,   255], [0,   255, 255], [255, 128, 0  ],
    [128, 0,   255], [0,   128, 255], [128, 128, 0  ], [0,   128, 128],
    [128, 0,   128], [192, 192, 192], [64,  64,  64 ], [255, 192, 203],
    [255, 165, 0  ],
], dtype=np.uint8)


def colorize_mask(mask, num_classes):
    pal = PALETTE
    if num_classes > len(pal):
        extra = np.random.RandomState(0).randint(0, 255,
                    size=(num_classes - len(pal), 3), dtype=np.uint8)
        pal = np.vstack([pal, extra])
    return pal[mask % len(pal)]


def predict_full_image(model, image_path, patch_size, num_classes, output_path):
    """滑窗推理，结果以半透明颜色叠加原图保存。"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取: {image_path}")

    rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for y in tqdm(range(0, h, patch_size), desc="滑窗推理", leave=False):
            for x in range(0, w, patch_size):
                patch = rgb[y:y + patch_size, x:x + patch_size]
                ph, pw = patch.shape[:2]
                if ph < patch_size or pw < patch_size:
                    pad = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                    pad[:ph, :pw] = patch
                    patch = pad
                t   = torch.from_numpy(patch.astype(np.float32) / 255.0)
                t   = t.permute(2, 0, 1).unsqueeze(0).to(device)
                out = model(t)
                p   = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
                pred_mask[y:y + ph, x:x + pw] = p[:ph, :pw]

    color   = colorize_mask(pred_mask, num_classes)
    overlay = cv2.addWeighted(rgb, 0.6, color, 0.4, 0)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"预测结果已保存至: {output_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    # ==================== 配置参数（按实际情况修改） ====================
    # patch 图像目录（prepare_dota_masks.py 输出的 image/）
    image_dir = r"/\GEE\BigEarthNet_dataset\bigearthnet_output_2000\image"

    # patch 掩码目录（prepare_dota_masks.py 输出的 masks/）
    mask_dir = r"/\GEE\BigEarthNet_dataset\bigearthnet_output_2000\mask"

    # classes.json 文件路径
    classes_json = r"D:\Pycharm Projects\torch\GEE\merged_industrial\classes.json"

    # 网络输入尺寸 (H, W)，必须与预处理时的 patch_size 一致
    input_size = (512, 512)

    # 批大小（根据显存调整）
    batch_size = 4              # RTX 5060 12G 建议先用 4，稳定后可试 8

    # 最大训练轮数（早停触发后会提前结束）
    epochs = 50

    # 学习率（Decoder 学习率；Encoder 自动 ×0.1）
    learning_rate = 5e-4

    # ImageNet 预训练权重
    pretrained = True           # 强烈推荐 True

    # 模型保存路径
    model_save = "unet_effb0_best_BEN_2000.pth"

    # 输出目录
    output_dir = "./output_effb0_BEN_2000"

    # 断点续训路径（不需要续训时设为 None）
    resume = None               # 例如：resume = "unet_effb0_best.pth"

    # 是否自动计算类别权重（数据集严重不平衡时建议开启）
    use_class_weights = False

    # ── 早停参数（移植自 unet_replace_SVD.py） ──────────────────
    # Val Loss 连续 patience 轮未改善则停止训练（推荐 5~15）
    patience = 15
    # ================================================================

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}")

    with open(classes_json, "r", encoding="utf-8") as f:
        info = json.load(f)
    # 如果 info 是列表（如 ["industrial"]），则将其视为 classes 列表
    if isinstance(info, list):
        classes = info
        num_classes = len(classes) + 1  # +1 是背景
    elif isinstance(info, dict):
        # 优先使用 "num_classes" 键，若没有则从 "classes" 长度推断
        if "num_classes" in info:
            num_classes = info["num_classes"]
        else:
            classes = info.get("classes", [])
            num_classes = len(classes) + 1
        classes = info.get("classes", [])
    else:
        raise ValueError("无法解析 classes.json 格式")
    class_names = ["background"] + classes

    # ── 模型 ────────────────────────────────────────────────────────
    model = EfficientUNet(num_classes=num_classes, pretrained=pretrained).to(device)
    print(f"已创建 EfficientUNet（EfficientNet-B0 Encoder, pretrained={pretrained}）")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"总参数量: {total_params:.2f} M")

    # ── 数据 ────────────────────────────────────────────────────────
    train_data, val_data = create_data_from_directory(image_dir, mask_dir)

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_data[1], num_classes)

    train_loader, val_loader = create_data_loaders(
        train_data, val_data, input_size=input_size, batch_size=batch_size
    )

    # ── 训练 ────────────────────────────────────────────────────────
    history = train_unet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=model_save,
        output_dir=output_dir,
        class_weights=class_weights,
        resume=resume,
        patience=patience,          # ← 早停参数传入
    )

    visualize_results(history, output_dir)
    print(f"模型已保存至: {model_save}")

    # ── 示例推理（加载最优 checkpoint） ─────────────────────────────
    if train_data[0]:
        ckpt = torch.load(model_save, map_location=device)
        model.load_state_dict(ckpt["model"])
        out_path = os.path.join(output_dir, "sample_prediction.png")
        try:
            predict_full_image(model, train_data[0][0], input_size[0], num_classes, out_path)
        except Exception as e:
            print(f"推理时出错: {e}")


if __name__ == "__main__":
    main()