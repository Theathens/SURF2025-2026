"""
merge_datasets.py
==================
SURF 2026 | T2 | 数据准备脚本

将多个数据源的 image/ mask/ 目录合并到统一的输出目录，
供 train_unet_industrial.py 直接读取。

合并逻辑：
    - 按来源前缀重命名（避免不同数据集同名文件互相覆盖）
    - 统计各来源的类别分布，输出到 merge_stats.json
    - 生成统一的 classes.json

支持的来源:
    bigearthnet/  ← BigEarthNet 2000 张 128x128
    gee_sliced/   ← GEE 大图切分后的子 patch

使用方法（已硬编码路径，直接运行即可）：
    python merge_datasets.py
"""

import os
import glob
import json
import shutil
import numpy as np
from PIL import Image


# ================================================================
# 硬编码配置（请根据你的实际路径修改！）
# ================================================================
SOURCES = [
    r"D:/Pycharm Projects/SURF-2026-0656/GEE/BigEarthNet_dataset/bigearthnet_output_5000",          # BigEarthNet 根目录
    r"D:/Pycharm Projects/SURF-2026-0656/GEE/GEE_dataset/test_ruhr_industrial/gee_output_ruhr_industrial/128x128_sliced",  # GEE 切片根目录
]
PREFIXES = ["ben", "gee"]          # 与 SOURCES 一一对应
OUT_DIR = r"//GEE/merged_industrial_3"  # 合并后输出目录
USE_SYMLINK = False                # True 为符号链接（仅 Linux），False 为复制文件
# ================================================================


def count_pixels(mask_dir, max_samples=500):
    """随机抽样统计各类像素数量"""
    files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    sample = files[:max_samples] if len(files) > max_samples else files
    counts = {}
    for f in sample:
        arr = np.array(Image.open(f).convert("L"))
        vals, cnts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            counts[v] = counts.get(v, 0) + c
    return counts


def main():
    assert len(SOURCES) == len(PREFIXES), "SOURCES 和 PREFIXES 数量必须一致"

    out_image = os.path.join(OUT_DIR, "image")
    out_mask  = os.path.join(OUT_DIR, "mask")
    os.makedirs(out_image, exist_ok=True)
    os.makedirs(out_mask,  exist_ok=True)

    stats = {}
    total_files = 0

    for src_dir, prefix in zip(SOURCES, PREFIXES):
        img_dir  = os.path.join(src_dir, "image")
        mask_dir = os.path.join(src_dir, "mask")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"⚠️  {src_dir} 缺少 image/ 或 mask/ 子目录，跳过")
            continue

        files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
        print(f"\n[{prefix}] 发现 {len(files)} 个 patch，来源: {src_dir}")

        # 统计像素分布
        pixel_counts = count_pixels(mask_dir)
        total_px = sum(pixel_counts.values())
        fg_ratio = pixel_counts.get(1, 0) / total_px if total_px > 0 else 0
        print(f"  背景占比: {(1-fg_ratio):.1%}, 工业像素占比: {fg_ratio:.1%}")
        stats[prefix] = {"n_patches": len(files), "fg_ratio": round(fg_ratio, 4),
                         "pixel_counts": pixel_counts}

        # 复制/链接文件
        for fname in files:
            new_name   = f"{prefix}_{fname}"
            src_img    = os.path.join(img_dir,  fname)
            src_mask   = os.path.join(mask_dir, fname)
            dst_img    = os.path.join(out_image, new_name)
            dst_mask   = os.path.join(out_mask,  new_name)

            if os.path.exists(dst_img):
                continue

            if USE_SYMLINK:
                os.symlink(os.path.abspath(src_img),  dst_img)
                os.symlink(os.path.abspath(src_mask), dst_mask)
            else:
                shutil.copy2(src_img,  dst_img)
                shutil.copy2(src_mask, dst_mask)

        total_files += len(files)
        print(f"  ✅ 已{'链接' if USE_SYMLINK else '复制'} {len(files)} 对文件")

    # 生成 classes.json
    classes_info = {
        "classes": ["industrial"],
        "class_to_id": {"industrial": 1},
        "num_classes": 2,
    }
    with open(os.path.join(OUT_DIR, "classes.json"), "w") as f:
        json.dump(classes_info, f, indent=2)

    # 生成 merge_stats.json
    stats["total_patches"] = total_files
    with open(os.path.join(OUT_DIR, "merge_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 总体工业像素占比
    total_fg   = sum(s["pixel_counts"].get(1, 0) for s in stats.values() if isinstance(s, dict) and "pixel_counts" in s)
    total_all  = sum(sum(s["pixel_counts"].values()) for s in stats.values() if isinstance(s, dict) and "pixel_counts" in s)
    overall_fg = total_fg / total_all if total_all > 0 else 0

    print(f"\n{'='*50}")
    print(f"合并完成: 共 {total_files} 对 patch → {OUT_DIR}")
    print(f"整体工业像素占比: {overall_fg:.2%}")
    if overall_fg < 0.05:
        print("⚠️  前景极度稀少，训练时务必开启 use_class_weights=True")
    print(f"classes.json 已写入: {OUT_DIR}/classes.json")
    print(f"merge_stats.json 已写入: {OUT_DIR}/merge_stats.json")


if __name__ == "__main__":
    main()