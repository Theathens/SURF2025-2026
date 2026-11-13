#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_run_unet_inference.py
============================
对 step2 下载的所有 Sentinel-2 影像批量运行 U-Net 推理，
输出每张影像的工业像素覆盖率。
修复：使用 PIL 读取图像以支持中文路径。
"""

import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# 添加你的模型定义路径（根据实际情况调整）
sys.path.append(r"/\GEE")
from train_unet_effb0_industrial import EfficientUNet

# ================= 硬编码配置 =================
IMAGE_DIR = r"/\confrontation\data\sentinel2_aoi\image"
OUTPUT_CSV = r"D:\Pycharm Projects\torch\confrontation\data\results\inference_results.csv"
MODEL_PATH = r"/\GEE\output_industrial_combine_GEE_BEN_3\unet_industrial_combine_GEE_BEN_3_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (128, 128)
# =============================================

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


def load_model(model_path: str):
    model = EfficientUNet(num_classes=2, pretrained=False).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ 模型已加载: {model_path}")
    return model


def predict_single_image(model, image_path: str) -> float:
    """使用 PIL 读取图像以支持中文路径"""
    try:
        img_pil = Image.open(image_path)
        img = np.array(img_pil.convert("RGB"))
    except Exception as e:
        raise ValueError(f"无法读取: {image_path}") from e

    if img.shape[:2] != INPUT_SIZE:
        img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    t = t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(t)
        pred = torch.argmax(output, dim=1).cpu().numpy()
    industrial_pixels = (pred == 1).sum()
    total_pixels = INPUT_SIZE[0] * INPUT_SIZE[1]
    return float(industrial_pixels / total_pixels)


def main():
    model = load_model(MODEL_PATH)
    image_files = sorted(Path(IMAGE_DIR).glob("*.png"))
    results = []

    print(f"找到 {len(image_files)} 张影像")
    for img_path in image_files:
        stem = img_path.stem
        parts = stem.split('_')
        year_str = parts[-1]
        if year_str.isdigit():
            year = int(year_str)
            facility_name = '_'.join(parts[:-1])
        else:
            facility_name = stem
            year = None

        try:
            coverage = predict_single_image(model, str(img_path))
            results.append({
                "image_file": img_path.name,
                "facility_name": facility_name,
                "year": year,
                "industrial_coverage": coverage
            })
            print(f"  {img_path.name}: {coverage:.2%}")
        except Exception as e:
            print(f"  ❌ {img_path.name} 推理失败: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 推理结果已保存: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()