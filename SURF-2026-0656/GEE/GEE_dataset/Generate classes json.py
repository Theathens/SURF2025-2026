"""
generate_classes_json.py
=========================
SURF 2026 | T2 | Step1 收尾脚本 ①

扫描 gee_industrial_data_pipeline.py 导出的 mask/ 文件夹，
统计实际出现的类别值，生成 classes.json，
供 train_unet_effb0.py 直接读取使用。

为什么需要这一步？
─────────────────────────────────────────────────
train_unet_effb0.py 通过 classes.json 里的 num_classes 来决定
U-Net 输出头的通道数。你现在的 GEE 工业数据集只有 2 类
（0=背景, 1=工业用地），和 DOTA 数据集的 17 类完全不同，
必须重新生成一份匹配的 classes.json，不能直接复用旧的。

用法:
    python generate_classes_json.py --mask_dir ./gee_output/mask \\
                                     --out_path ./gee_output/classes.json
─────────────────────────────────────────────────
"""

import os
import json
import glob
import argparse

import numpy as np
from PIL import Image


def scan_mask_classes(mask_dir: str):
    """
    扫描 mask_dir 下所有 PNG，统计出现过的所有像素值（即类别索引）。
    返回排序后的类别值列表，例如 [0, 1]。
    """
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_files:
        raise ValueError(f"在 {mask_dir} 中未找到任何 .png 掩码文件")

    seen_values = set()
    pixel_counts = {}

    print(f"正在扫描 {len(mask_files)} 个掩码文件...")
    for i, fpath in enumerate(mask_files):
        with Image.open(fpath) as img:
            arr = np.array(img)
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            seen_values.add(v)
            pixel_counts[v] = pixel_counts.get(v, 0) + c

        if (i + 1) % 50 == 0 or (i + 1) == len(mask_files):
            print(f"  已扫描 {i + 1}/{len(mask_files)}")

    return sorted(seen_values), pixel_counts


def main():
    parser = argparse.ArgumentParser(description="生成 GEE 工业数据集的 classes.json")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="GEE 导出的掩码文件夹路径")
    parser.add_argument("--out_path", type=str, required=True,
                        help="classes.json 输出路径")
    parser.add_argument("--class_names", type=str, nargs="*", default=None,
                        help="非背景类别的名称（按类别值从小到大排列，"
                             "不含背景）。默认单类时为 ['industrial']")
    args = parser.parse_args()

    values, pixel_counts = scan_mask_classes(args.mask_dir)
    print(f"\n发现的类别值: {values}")

    total_pixels = sum(pixel_counts.values())
    for v in values:
        ratio = pixel_counts[v] / total_pixels * 100
        print(f"  值={v}: {pixel_counts[v]:,} 像素 ({ratio:.2f}%)")

    # 0 默认是背景，不计入 classes 列表（与 train_unet_effb0.py 的
    # SegmentationDataset 约定一致：0=背景，1..N=类别）
    non_bg_values = [v for v in values if v != 0]

    if args.class_names:
        if len(args.class_names) != len(non_bg_values):
            raise ValueError(
                f"提供的 class_names 数量 ({len(args.class_names)}) "
                f"与检测到的非背景类别数 ({len(non_bg_values)}) 不一致"
            )
        class_names = args.class_names
    else:
        # 默认命名：单类时叫 industrial，多类时按值编号
        if len(non_bg_values) == 1:
            class_names = ["industrial"]
        else:
            class_names = [f"class_{v}" for v in non_bg_values]

    class_to_id = {name: int(v) for name, v in zip(class_names, non_bg_values)}
    num_classes = len(class_names) + 1   # +1 背景

    result = {
        "classes": class_names,
        "class_to_id": class_to_id,
        "num_classes": num_classes,
    }

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nclasses.json 已生成: {args.out_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 不平衡程度提示
    if 0 in pixel_counts and non_bg_values:
        bg_ratio = pixel_counts[0] / total_pixels
        if bg_ratio > 0.9:
            fg_ratio = 1 - bg_ratio
            print(f"\n⚠️  类别极度不平衡：背景占 {bg_ratio:.1%}，"
                  f"前景仅占 {fg_ratio:.1%}")
            print("    强烈建议在 train_unet_effb0.py 配置中设置:")
            print("        use_class_weights = True")


if __name__ == "__main__":
    main()