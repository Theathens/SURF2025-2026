import os
import json
import numpy as np
import cv2
import torch
from torch import nn
from train_unet_effb0 import EfficientUNet  # 导入你的模型定义
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. 加载 T4 的伪造数据
# ============================================================
def load_fake_tile(npy_path, json_path):
    """读取伪造影像及其元数据"""
    with open(json_path, 'r') as f:
        meta = json.load(f)
    data = np.load(npy_path)  # 假设 shape 为 (H, W, C) 或 (C, H, W)
    # 统一转为 (C, H, W) 以便操作
    if data.ndim == 3 and data.shape[-1] in [3, 6]:
        data = np.transpose(data, (2, 0, 1))  # HWC -> CHW
    return data, meta


# ============================================================
# 2. 适配器：将 6 通道 30m 数据转为 3 通道 10m 规模
# ============================================================
def preprocess_for_unet(fake_tile, target_size=(512, 512)):
    """
    输入: (C, H, W) 的 numpy 数组，C=6
    输出: (1, 3, 512, 512) 的 SURF-2026-0656 Tensor，数值严格在 [0, 1] 之间
    """
    # 步骤 A: 只取前 3 个波段 (RGB)
    rgb = fake_tile[:3, :, :]  # (3, H, W)

    # 步骤 B: 转为 HWC 以便 OpenCV 调整大小
    rgb_hwc = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

    # 步骤 C: 重采样到 512x512
    resized = cv2.resize(rgb_hwc, target_size, interpolation=cv2.INTER_LINEAR)

    # ========== 🔥 关键修复：强制归一化到 [0, 1] ==========
    # 无论 T4 给的是 [-1,1] 还是 [0,255]，都强行映射到 [0,1]
    min_val = resized.min()
    max_val = resized.max()

    # 如果数据已经是 [0,1] 但带有微小噪声，直接 clip 即可
    if max_val > 1.0 or min_val < 0.0:
        # 如果极差为 0（全黑/全灰），直接返回全 0
        if max_val - min_val < 1e-6:
            print("  警告：检测到无效恒定值，返回全零张量")
            return torch.zeros((1, 3, target_size[0], target_size[1]))
        # 线性拉伸到 [0, 1]
        resized = (resized - min_val) / (max_val - min_val)
    else:
        # 已经在 [0,1] 内，只需保证无溢出
        resized = np.clip(resized, 0.0, 1.0)

    # 转回 CHW
    tensor = torch.from_numpy(np.transpose(resized, (2, 0, 1))).float()
    return tensor.unsqueeze(0)  # (1, 3, 512, 512)


# ============================================================
# 3. 主评估流程
# ============================================================
def evaluate_redteam(satellite_dir, model_path, device):
    # 加载你的最优模型
    model = EfficientUNet(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    results = []

    # 遍历 satellite 文件夹
    for file in os.listdir(satellite_dir):
        if not file.endswith('.npy'):
            continue

        npy_path = os.path.join(satellite_dir, file)
        json_path = npy_path.replace('.npy', '_metadata.json')

        if not os.path.exists(json_path):
            print(f"跳过 {file}，缺少元数据")
            continue

        print(f"正在处理对抗样本: {file}")
        data, meta = load_fake_tile(npy_path, json_path)

        # 预处理
        input_tensor = preprocess_for_unet(data).to(device)

        # 推理（无梯度）
        with torch.no_grad():
            output = model(input_tensor)  # (1, 2, 512, 512)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (512, 512)

        # 统计“工业用地”占比（类别 1 的像素比例）
        industrial_ratio = np.mean(pred_mask == 1)

        # 提取攻击类型和强度
        # 例如文件名: SAT_POWER_DAILY_TEMPORAL_SHIFT_LOW_06.npy
        name_parts = file.split('_')
        # 简单的粗暴提取逻辑（你可根据实际调整）
        attack_type = name_parts[2] + '_' + name_parts[3] if len(name_parts) > 4 else 'unknown'
        severity = name_parts[4] if len(name_parts) > 4 else 'unknown'

        results.append({
            'filename': file,
            'tile_id': meta.get('tile_id'),
            'attack_type': attack_type,
            'severity': severity,
            'industrial_ratio': industrial_ratio,
            'bands': meta.get('bands'),
            'resolution': meta.get('resolution_m')
        })

        # 可选：保存一张预测叠加图（用于可视化报告）
        # plt.imsave(f"pred_{file.replace('.npy','.png')}", pred_mask, cmap='jet')

    return pd.DataFrame(results)


# ============================================================
# 4. 执行 & 输出报告
# ============================================================
if __name__ == "__main__":
    # 配置路径（改成你的实际路径）
    SATELLITE_DIR = "C:/Users/25145/Desktop/SURF2026/T2 vs T4/redteam_delivery/satellite"  # 放 .npy 的地方
    MODEL_PATH = "//GEE/unet_effb0_best_ruhr_industrial.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = evaluate_redteam(SATELLITE_DIR, MODEL_PATH, DEVICE)

    # 打印汇总统计（这就是你的对抗测试报告核心数据）
    print("\n=== 红蓝对抗测试报告 ===")
    print(df.groupby(['attack_type', 'severity'])['industrial_ratio'].mean())

    # 保存为 Excel，方便交给 T5 或导师
    df.to_excel("redteam_robustness_report.xlsx", index=False)
    print("\n详细结果已保存至 redteam_robustness_report.xlsx")