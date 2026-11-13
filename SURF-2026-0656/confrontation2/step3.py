#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_run_unet_inference.py (重构版)
=====================================
功能：
  1. 读取 step2 下载的所有 PNG 影像（任意尺寸）
  2. 使用预训练的 U-Net 模型（二分类：工业/非工业）进行推理
  3. 输出每张影像的工业像素占比和绝对工业像素数
  4. 自动提取文件名中的厂区名和年份，以便后续配对

输入：sentinel2_aoi/image/ 目录下的所有 PNG
输出：inference_results.csv（含 image_file, facility_name, year, industrial_ratio, industrial_pixels）

依赖：SURF-2026-0656, torchvision, PIL, pandas, numpy, opencv-python
"""

import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# 添加模型定义路径（确保可以导入你的模型类）
sys.path.append(r"/\GEE")  # 这里替换成你的模型文件所在目录
from train_unet_effb0_industrial import EfficientUNet  # 假设模型类名是 EfficientUNet

# ================= 硬编码配置 =================
IMAGE_DIR = r"/\confrontation2\data\01_send_to_group3_PUBLIC_BLIND\sentinel2_aoi\image"
OUTPUT_CSV = r"D:\Pycharm Projects\torch\confrontation2\data\01_send_to_group3_PUBLIC_BLIND\results\inference_results.csv"
MODEL_PATH = r"/\GEE\output_industrial_combine_GEE_BEN_3\unet_industrial_combine_GEE_BEN_3_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (128, 128)  # 模型期望的输入尺寸
# =============================================

# 创建输出目录
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


def load_model(model_path: str):
    """加载模型"""
    model = EfficientUNet(num_classes=2, pretrained=False).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)  # 兼容不同保存格式
    model.load_state_dict(state)
    model.eval()
    print(f"✅ 模型已加载: {model_path}")
    return model


def predict_single_image(model, image_path: str) -> dict:
    """
    对单张影像进行推理，返回工业像素占比和绝对像素数。
    影像会缩放至 INPUT_SIZE，但输出结果按原始图像像素数归一化。
    """
    # 读取图像 (PIL 支持中文路径)
    img_pil = Image.open(image_path)
    img = np.array(img_pil.convert("RGB"))
    h, w = img.shape[:2]

    # 调整尺寸到模型输入尺寸
    if (h, w) != INPUT_SIZE:
        img_resized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    # 归一化并转为 tensor
    t = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(t)
        pred = torch.argmax(output, dim=1).cpu().numpy()  # (1, H, W)

    # 在缩放后的图像上统计工业像素比例
    total_pixels_resized = INPUT_SIZE[0] * INPUT_SIZE[1]
    industrial_pixels_resized = (pred == 1).sum()
    ratio_resized = float(industrial_pixels_resized / total_pixels_resized)

    # 如果希望按原始尺寸估算绝对工业像素数，可以等比例放大：
    # 但这只是一个近似，更好的做法是不缩放直接推理（但模型固定尺寸，只能这样）
    # 我们输出缩放后的比例作为最终指标（因为所有图像缩放至统一尺寸，比例具有可比性）
    # 同时返回绝对工业像素数（按缩放后的尺寸计）
    return {
        "industrial_ratio": ratio_resized,
        "industrial_pixels_scaled": int(industrial_pixels_resized),
        "image_shape": (h, w)
    }


def main():
    # 加载模型
    model = load_model(MODEL_PATH)

    # 获取所有 PNG 文件
    image_files = sorted(Path(IMAGE_DIR).glob("*.png"))
    print(f"找到 {len(image_files)} 张影像")

    results = []
    for img_path in image_files:
        stem = img_path.stem
        # 解析文件名：厂区名_年份.png（厂区名可能包含下划线，但年份是最后一部分）
        parts = stem.split('_')
        if parts[-1].isdigit():
            year = int(parts[-1])
            facility_name = '_'.join(parts[:-1])
        else:
            # 如果最后一部分不是数字，则整个文件名作为厂区名，年份缺失
            facility_name = stem
            year = None

        try:
            pred_dict = predict_single_image(model, str(img_path))
            results.append({
                "image_file": img_path.name,
                "facility_name": facility_name,
                "year": year,
                "industrial_ratio": round(pred_dict["industrial_ratio"], 6),
                "industrial_pixels_scaled": pred_dict["industrial_pixels_scaled"],
            })
            print(f"  {img_path.name}: 工业占比 {pred_dict['industrial_ratio']:.2%}")
        except Exception as e:
            print(f"  ❌ {img_path.name} 推理失败: {e}")

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 推理结果已保存至: {OUTPUT_CSV}")
    print(f"   共处理 {len(df)} 张影像")


if __name__ == "__main__":
    main()