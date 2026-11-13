"""
slice_gee_patches.py
=====================
SURF 2026 | T2 | 数据准备脚本

将 GEE 导出的大图（约 377×574）切分成与 BigEarthNet 相同尺寸
的 128×128 子 patch，方便两个数据集合并训练。

为什么不直接 resize？
────────────────────────────────────────────────────
把 377×574 的鲁尔区大图直接 resize 到 128×128，
相当于把 29km × 46km 的区域压缩成一张邮票。
以 10m/像素的 Sentinel-2 分辨率计算，一个 128×128 的 patch
已经覆盖约 1.28km × 1.28km 的地面面积，
这才是模型"看一眼"能理解的合理尺度。

输出结构（与 BigEarthNet 输出完全相同，可直接混合）：
    <out_dir>/
        image/    gee_patch_XXXX_rYYY_cZZZ.png   ← 子图像
        mask/     gee_patch_XXXX_rYYY_cZZZ.png   ← 对应掩码

用法:
    python slice_gee_patches.py \\
        --image_dir ./gee_output/image \\
        --mask_dir  ./gee_output/mask \\
        --out_dir   ./gee_sliced \\
        --patch_size 128 \\
        --keep_empty_ratio 0.1
────────────────────────────────────────────────────
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image


def slice_one_image(img_path, mask_path, out_image_dir, out_mask_dir,
                    patch_size=128, keep_empty_ratio=0.1, rng=None):
    """
    把一张大图和对应掩码切成 patch_size × patch_size 的子图，
    保留含前景（工业用地）的子图和一定比例的纯背景子图。

    返回 (n_kept, n_skipped)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    img_arr  = np.array(Image.open(img_path).convert("RGB"))
    mask_arr = np.array(Image.open(mask_path).convert("L"))

    H, W = img_arr.shape[:2]
    base  = os.path.splitext(os.path.basename(img_path))[0]

    kept = skipped = 0

    for row in range(0, H - patch_size + 1, patch_size):
        for col in range(0, W - patch_size + 1, patch_size):
            img_patch  = img_arr[ row:row+patch_size, col:col+patch_size, :]
            mask_patch = mask_arr[row:row+patch_size, col:col+patch_size]

            has_fg = bool((mask_patch > 0).any())

            # 纯背景 patch 按比例随机保留
            if not has_fg and rng.rand() > keep_empty_ratio:
                skipped += 1
                continue

            fname = f"gee_{base}_r{row:04d}_c{col:04d}.png"
            Image.fromarray(img_patch,  mode="RGB").save(
                os.path.join(out_image_dir, fname))
            Image.fromarray(mask_patch, mode="L").save(
                os.path.join(out_mask_dir, fname))
            kept += 1

    return kept, skipped


def main():
    parser = argparse.ArgumentParser(description="将GEE大图切分成128x128子patch")
    parser.add_argument("--image_dir",         type=str, required=True)
    parser.add_argument("--mask_dir",          type=str, required=True)
    parser.add_argument("--out_dir",           type=str, required=True)
    parser.add_argument("--patch_size",        type=int, default=128)
    parser.add_argument("--keep_empty_ratio",  type=float, default=0.1,
                        help="纯背景子图的保留比例（0~1），工业区域很稀疏时建议 ≤0.1")
    args = parser.parse_args()

    out_image_dir = os.path.join(args.out_dir, "image")
    out_mask_dir  = os.path.join(args.out_dir, "mask")
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir,  exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    rng = np.random.RandomState(42)

    total_kept = total_skipped = 0
    print(f"共找到 {len(img_files)} 张大图，开始切分...")

    for i, img_path in enumerate(img_files, 1):
        mask_path = os.path.join(args.mask_dir, os.path.basename(img_path))
        if not os.path.exists(mask_path):
            print(f"  [{i}] ⚠️ 无对应掩码，跳过: {os.path.basename(img_path)}")
            continue

        kept, skipped = slice_one_image(
            img_path, mask_path,
            out_image_dir, out_mask_dir,
            patch_size=args.patch_size,
            keep_empty_ratio=args.keep_empty_ratio,
            rng=rng,
        )
        total_kept    += kept
        total_skipped += skipped

        if i % 20 == 0 or i == len(img_files):
            print(f"  [{i}/{len(img_files)}] 累计保留 {total_kept} 个，丢弃 {total_skipped} 个纯背景 patch")

    print(f"\n完成! 共生成 {total_kept} 个子 patch → {args.out_dir}")
    print(f"（纯背景子 patch 按 keep_empty_ratio={args.keep_empty_ratio} 保留了"
          f" {int(total_kept/(total_kept+total_skipped+1e-8)*100)}%）")


if __name__ == "__main__":
    import sys
    sys.argv = [
        "slice_gee_patches.py",
        "--image_dir", "D:/Pycharm Projects/SURF-2026-0656/GEE/GEE_dataset/test_ruhr_industrial/gee_output_ruhr_industrial/image",
        "--mask_dir", "D:/Pycharm Projects/SURF-2026-0656/GEE/GEE_dataset/test_ruhr_industrial/gee_output_ruhr_industrial/mask",
        "--out_dir", "D:/Pycharm Projects/SURF-2026-0656\GEE\GEE_dataset/test_ruhr_industrial/gee_output_ruhr_industrial/128x128_sliced",
        "--patch_size", "128",
        "--keep_empty_ratio", "0.1"
    ]
    main()