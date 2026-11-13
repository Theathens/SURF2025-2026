"""
zonal_stats.py
==============
SURF 2026 | T2 | Week 5-6 核心交付物

功能：
  1. 用训练好的 U-Net 对 GEE 卫星影像 patch 做推理，
     得到每张图的工业设施边界掩码
  2. 在掩码内计算三类活动指标:
       · 植被抑制指数  (VDI)  ← 植被越少说明工业越活跃
       · 亮度活跃指数  (BAI)  ← 工业设施反射率高于背景
       · 纹理复杂指数  (TCI)  ← 工业设施内部纹理比自然区域复杂
       · 综合活动指数  (CAI)  ← 三者加权，0=完全非活跃，1=高强度运行
  3. 如果提供了 activity_features.tif（包含 NDVI/夜光/温度），
     额外计算基于真实光谱的活动指标
  4. 输出三个 Sheet 的 Excel 报告，交给 T5 分析师

所有路径均在下方「配置区」硬编码，无需命令行参数。
直接在 PyCharm 里点击运行即可。
"""

import os
import sys
import glob
import json
import warnings
import importlib.util
from datetime import datetime

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch

warnings.filterwarnings("ignore")

# ============================================================
# ★★★ 配置区：修改这里的路径，其他代码不需要改动 ★★★
# ============================================================

# GEE 导出的原始卫星影像目录（128×128 PNG，RGB）
IMAGE_DIR = r"/\GEE\GEE_dataset\test_ruhr_industrial\gee_output_ruhr_industrial\128x128_sliced\image"

# 对应的真实标签掩码目录（用于计算"真实边界内"的统计，可不填）
# 填 None 则跳过真实标签对比，只用 U-Net 预测掩码
GT_MASK_DIR = r"/\GEE\GEE_dataset\test_ruhr_industrial\gee_output_ruhr_industrial\128x128_sliced\mask"

# 训练好的 U-Net 工业分割模型权重
MODEL_PATH = r"/\GEE\output_industrial_combine_GEE_BEN_3\unet_industrial_combine_GEE_BEN_3_best.pth"

# train_unet_industrial.py 的路径（用于动态 import EfficientUNet 类）
TRAIN_SCRIPT_PATH = r"/\GEE\train_unet_effb0_industrial.py"

# classes.json（用于获取 num_classes）
CLASSES_JSON = r"D:\Pycharm Projects\torch\GEE\merged_industrial_3\classes.json"

# （可选）activity_features.tif：GEE 导出的 [NDVI, nightlight, temperature_C] 三波段文件
# 如果没有下载成功，填 None，脚本会自动跳过光谱指标计算
ACTIVITY_FEATURES_TIF = r"D:\Pycharm Projects\torch\GEE\GEE_dataset\test_ruhr_industrial\gee_output_ruhr_industrial\Ruhr_Activity_Features_30m.tif"

# 输出目录（自动创建）
OUTPUT_DIR = r"//GEE/zonal_stats_output"

# 模型输入尺寸（与训练时一致）
INPUT_SIZE = (128, 128)

# 一次推理的批大小（显存不够时调小）
BATCH_SIZE = 16

# 是否保存每张图的可视化对比图（影像 | 预测掩码 | 活动指数热力图）
SAVE_VIS = True
VIS_MAX_IMAGES = 30   # 最多保存多少张，避免输出目录过大

# ============================================================
# 以下代码无需修改
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 动态加载 EfficientUNet ───────────────────────────────────
def load_model_class(train_script_path: str):
    spec   = importlib.util.spec_from_file_location("train_module", train_script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EfficientUNet


def load_unet(model_path: str, num_classes: int, train_script_path: str):
    EfficientUNet = load_model_class(train_script_path)
    model = EfficientUNet(num_classes=num_classes, pretrained=False).to(device)
    ckpt  = torch.load(model_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"已加载模型: {model_path}")
    return model


# ── 活动指数计算 ──────────────────────────────────────────────
def compute_activity_indices(rgb: np.ndarray, mask: np.ndarray):
    """
    从 RGB 图像和二值掩码计算工业活动指数。

    rgb:  (H, W, 3) uint8，像素值 0-255
    mask: (H, W)    uint8，1=工业区，0=背景

    返回: dict，包含各项指标和综合活动指数
    """
    H, W = rgb.shape[:2]
    region = mask.astype(bool)
    background = ~region

    if region.sum() == 0:
        # 掩码全空：返回 NaN
        return {
            "industrial_pixel_count":  0,
            "industrial_coverage_pct": 0.0,
            "vdi_industrial":   float("nan"),
            "vdi_background":   float("nan"),
            "bai_industrial":   float("nan"),
            "bai_background":   float("nan"),
            "tci_industrial":   float("nan"),
            "tci_background":   float("nan"),
            "composite_activity_index": float("nan"),
        }

    r = rgb[:, :, 0].astype(np.float32) / 255.0
    g = rgb[:, :, 1].astype(np.float32) / 255.0
    b = rgb[:, :, 2].astype(np.float32) / 255.0

    # ── 植被抑制指数 VDI：工业区植被越少，值越高 ─────────────
    # 植被像素通常 G >> R, B（绿色）；工业/裸地则相对 R 较高
    # VDI = 1 - green_dominance = 1 - (G - 0.5*(R+B)) / (G + 0.5*(R+B) + 1e-8)
    green_dom = (g - 0.5 * (r + b)) / (g + 0.5 * (r + b) + 1e-8)
    vdi = np.clip(1.0 - green_dom, 0, 1)
    vdi_ind = float(vdi[region].mean())
    vdi_bg  = float(vdi[background].mean()) if background.sum() > 0 else float("nan")

    # ── 亮度活跃指数 BAI：工业建筑屋顶/停车场比自然区域更亮 ──
    brightness = (r + g + b) / 3.0
    bai_ind = float(brightness[region].mean())
    bai_bg  = float(brightness[background].mean()) if background.sum() > 0 else float("nan")

    # ── 纹理复杂指数 TCI：工业设施内部结构比草地/农田复杂 ─────
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # 局部标准差（用 3×3 窗口的方差近似纹理复杂度）
    gray_sq  = cv2.blur(gray * gray, (3, 3))
    gray_mu  = cv2.blur(gray,        (3, 3))
    variance = np.clip(gray_sq - gray_mu ** 2, 0, None)
    tci = np.sqrt(variance)
    tci_ind = float(tci[region].mean())
    tci_bg  = float(tci[background].mean()) if background.sum() > 0 else float("nan")

    # ── 综合活动指数 CAI ──────────────────────────────────────
    # 权重: VDI 0.40（最关键，植被与工业对立），
    #        BAI 0.35（亮度直接对应地面覆被），
    #        TCI 0.25（纹理区分精细结构）
    # 每项先相对背景归一化，使指标含义为"工业区比背景活跃多少"
    def relative(ind_val, bg_val, clip_max=2.0):
        if np.isnan(bg_val) or bg_val == 0:
            return float(np.clip(ind_val, 0, 1))
        return float(np.clip(ind_val / (bg_val + 1e-8) - 1.0, 0, clip_max) / clip_max)

    cai = (
        relative(vdi_ind, vdi_bg) * 0.40 +
        relative(bai_ind, bai_bg) * 0.35 +
        relative(tci_ind, tci_bg) * 0.25
    )
    cai = float(np.clip(cai, 0.0, 1.0))

    return {
        "industrial_pixel_count":  int(region.sum()),
        "industrial_coverage_pct": round(region.sum() / (H * W) * 100, 2),
        "vdi_industrial":   round(vdi_ind, 4),
        "vdi_background":   round(vdi_bg,  4) if not np.isnan(vdi_bg)  else None,
        "bai_industrial":   round(bai_ind, 4),
        "bai_background":   round(bai_bg,  4) if not np.isnan(bai_bg)  else None,
        "tci_industrial":   round(tci_ind, 4),
        "tci_background":   round(tci_bg,  4) if not np.isnan(tci_bg)  else None,
        "composite_activity_index": round(cai, 4),
    }


def compute_spectral_stats(activity_tif: str, mask_full: np.ndarray):
    """
    如果提供了 activity_features.tif，从中提取 NDVI / 夜光 / 温度。
    mask_full 是与整个 AOI 大小相同的工业区掩码（由各 patch 掩码拼合而来）。
    """
    try:
        import rasterio
    except ImportError:
        print("  ⚠️  rasterio 未安装，跳过光谱指标计算")
        return {}

    if not os.path.exists(activity_tif):
        return {}

    with rasterio.open(activity_tif) as src:
        data = src.read()   # (3, H, W): [NDVI, nightlight, temperature_C]

    # 把 mask_full resize 到 tif 尺寸（可能不同）
    tif_h, tif_w = data.shape[1], data.shape[2]
    mask_rs = cv2.resize(mask_full.astype(np.uint8),
                         (tif_w, tif_h), interpolation=cv2.INTER_NEAREST).astype(bool)

    if mask_rs.sum() == 0:
        return {}

    ndvi      = data[0][mask_rs]
    nightlight = data[1][mask_rs]
    temp_c    = data[2][mask_rs]

    return {
        "ndvi_mean_industrial":        round(float(np.nanmean(ndvi)),       4),
        "ndvi_std_industrial":         round(float(np.nanstd(ndvi)),        4),
        "nightlight_mean_industrial":  round(float(np.nanmean(nightlight)), 4),
        "temperature_mean_industrial": round(float(np.nanmean(temp_c)),     4),
    }


# ── 推理 ────────────────────────────────────────────────────
def predict_batch(model, img_paths: list, input_size: tuple):
    """批量推理，返回 list of (pred_mask, rgb_img)"""
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(img_paths), BATCH_SIZE):
            batch_paths = img_paths[i:i + BATCH_SIZE]
            tensors, originals = [], []
            for p in batch_paths:
                img = cv2.imread(p)
                if img is None:
                    img = np.zeros((*input_size, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                if (h, w) != input_size:
                    img = cv2.resize(img, (input_size[1], input_size[0]),
                                     interpolation=cv2.INTER_LINEAR)
                originals.append(img)
                t = torch.from_numpy(img.astype(np.float32) / 255.0)
                tensors.append(t.permute(2, 0, 1))

            batch_t = torch.stack(tensors).to(device)
            outputs = model(batch_t)
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()  # (B, H, W)
            for pred, rgb in zip(preds, originals):
                results.append((pred.astype(np.uint8), rgb))
    return results


# ── 可视化 ───────────────────────────────────────────────────
def save_visualization(rgb: np.ndarray, pred_mask: np.ndarray,
                       gt_mask, cai: float, patch_name: str, out_path: str):
    """
    保存三联图：原始卫星影像 | 预测工业区掩码 | CAI 活动热力图
    """
    ncols = 3 if gt_mask is None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="Oranges", vmin=0, vmax=1)
    axes[1].set_title(f"U-Net 预测工业区\n覆盖 {pred_mask.mean()*100:.1f}%", fontsize=10)
    axes[1].axis("off")

    # CAI 热力图：在工业区内叠加亮度作为热力图
    brightness = rgb.mean(axis=2).astype(np.float32) / 255.0
    heat = brightness * pred_mask.astype(np.float32)
    axes[2].imshow(rgb, alpha=0.5)
    im = axes[2].imshow(heat, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title(f"活动强度热力图\nCAI = {cai:.3f}", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    if gt_mask is not None and ncols == 4:
        axes[3].imshow(gt_mask, cmap="Blues", vmin=0, vmax=1)
        agree = ((pred_mask == 1) & (gt_mask == 1)).sum()
        axes[3].set_title(f"真实标签\n与预测重叠 {agree} px", fontsize=10)
        axes[3].axis("off")

    fig.suptitle(patch_name, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ── Excel 报告 ───────────────────────────────────────────────
def write_excel_report(records: list, summary: dict, out_path: str):
    from openpyxl import Workbook
    from openpyxl.styles import (Font, PatternFill, Alignment,
                                  Border, Side)
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule

    wb = Workbook()

    hdr_fill  = PatternFill("solid", fgColor="1F4E79")
    hdr_font  = Font(color="FFFFFF", bold=True, name="Arial", size=10)
    thin      = Border(left=Side("thin"), right=Side("thin"),
                       top=Side("thin"),  bottom=Side("thin"))
    center    = Alignment(horizontal="center", vertical="center")

    # ══════════════════════════════════════════════════
    # Sheet 1：逐 patch 活动指数（交给 T5 的主要数据）
    # ══════════════════════════════════════════════════
    ws1 = wb.active
    ws1.title = "逐Patch活动指数"

    cols = [
        ("Patch 文件名",         "patch_name"),
        ("工业像素数",            "industrial_pixel_count"),
        ("工业覆盖率(%)",         "industrial_coverage_pct"),
        ("植被抑制指数(VDI)",      "vdi_industrial"),
        ("亮度活跃指数(BAI)",      "bai_industrial"),
        ("纹理复杂指数(TCI)",      "tci_industrial"),
        ("综合活动指数(CAI) ★",    "composite_activity_index"),
        ("背景VDI",               "vdi_background"),
        ("背景BAI",               "bai_background"),
        ("工业/背景 活跃比",       "active_ratio"),
        ("GT工业覆盖率(%)",        "gt_coverage_pct"),
        ("预测与GT重叠率",         "iou_with_gt"),
    ]

    for ci, (header, _) in enumerate(cols, 1):
        c = ws1.cell(row=1, column=ci, value=header)
        c.fill = hdr_fill; c.font = hdr_font
        c.alignment = center; c.border = thin
    ws1.row_dimensions[1].height = 32

    for ri, rec in enumerate(records, 2):
        # 计算衍生指标
        bai_ind = rec.get("bai_industrial") or 0
        bai_bg  = rec.get("bai_background") or 1
        rec["active_ratio"] = round(bai_ind / (bai_bg + 1e-8), 3)

        for ci, (_, key) in enumerate(cols, 1):
            val = rec.get(key, "")
            if val is None: val = ""
            c = ws1.cell(row=ri, column=ci, value=val)
            c.border = thin
            c.font = Font(name="Arial", size=10)
            c.alignment = center
            if key in ("composite_activity_index", "iou_with_gt",
                       "active_ratio", "vdi_industrial", "bai_industrial"):
                c.number_format = "0.0000"
            if key in ("industrial_coverage_pct", "gt_coverage_pct"):
                c.number_format = "0.00"

    # 对 CAI 列加绿→黄→红色阶（低活跃→高活跃）
    cai_col  = get_column_letter(7)
    last_row = len(records) + 1
    ws1.conditional_formatting.add(
        f"{cai_col}2:{cai_col}{last_row}",
        ColorScaleRule(start_type="min", start_color="63BE7B",
                       mid_type="percentile", mid_value=50, mid_color="FFEB84",
                       end_type="max", end_color="F8696B")
    )

    # 列宽
    widths = [40, 14, 14, 18, 18, 18, 20, 12, 12, 14, 16, 16]
    for i, w in enumerate(widths, 1):
        ws1.column_dimensions[get_column_letter(i)].width = w
    ws1.freeze_panes = "A2"

    # ══════════════════════════════════════════════════
    # Sheet 2：汇总统计（供 T5 直接引用的关键数字）
    # ══════════════════════════════════════════════════
    ws2 = wb.create_sheet("汇总统计")

    summary_rows = [
        ("生成时间",             summary.get("generated_at", "")),
        ("处理 Patch 总数",       summary.get("total_patches", "")),
        ("含工业区 Patch 数",     summary.get("patches_with_industrial", "")),
        ("工业区检测率",          f"{summary.get('detection_rate', 0):.1%}"),
        ("",                     ""),
        ("── 活动指数统计 ──",    ""),
        ("CAI 均值",              round(summary.get("cai_mean", 0), 4)),
        ("CAI 标准差",            round(summary.get("cai_std", 0),  4)),
        ("CAI 中位数",            round(summary.get("cai_median", 0), 4)),
        ("CAI 75百分位",          round(summary.get("cai_p75", 0), 4)),
        ("CAI 95百分位",          round(summary.get("cai_p95", 0), 4)),
        ("",                     ""),
        ("── 空间覆盖统计 ──",    ""),
        ("平均工业覆盖率",        f"{summary.get('mean_coverage', 0):.2f}%"),
        ("最大工业覆盖率",        f"{summary.get('max_coverage', 0):.2f}%"),
        ("总工业像素数",          summary.get("total_industrial_pixels", 0)),
        ("",                     ""),
        ("── 光谱指标（需activity_features.tif）──", ""),
        ("NDVI 均值（工业区内）", summary.get("ndvi_mean", "N/A（无TIF文件）")),
        ("夜间灯光均值（工业区）", summary.get("nightlight_mean", "N/A")),
        ("地表温度均值（工业区）", summary.get("temperature_mean", "N/A")),
    ]

    ws2["A1"] = "指标名称"
    ws2["B1"] = "数值"
    for c in [ws2["A1"], ws2["B1"]]:
        c.fill = hdr_fill; c.font = hdr_font; c.alignment = center

    for ri, (k, v) in enumerate(summary_rows, 2):
        ws2.cell(row=ri, column=1, value=k).font = Font(name="Arial", size=10)
        vc = ws2.cell(row=ri, column=2, value=v)
        vc.font = Font(name="Arial", size=10)
        vc.alignment = center

    ws2.column_dimensions["A"].width = 36
    ws2.column_dimensions["B"].width = 24

    # ══════════════════════════════════════════════════
    # Sheet 3：指标说明（给 T5 的解读指南）
    # ══════════════════════════════════════════════════
    ws3 = wb.create_sheet("指标说明")
    explanations = [
        ("指标缩写", "全称", "物理含义", "高值含义", "典型范围"),
        ("VDI", "植被抑制指数",
         "1 - 绿色主导程度。工业用地植被稀少，VDI高",
         "工业高度活跃/裸地", "0.3~0.8"),
        ("BAI", "亮度活跃指数",
         "RGB均值/255。工业屋顶、水泥地反射率高",
         "大面积硬质地面", "0.2~0.7"),
        ("TCI", "纹理复杂指数",
         "局部灰度标准差。工业厂房内部结构比草地复杂",
         "密集工业设施", "0.05~0.25"),
        ("CAI", "综合活动指数",
         "VDI×0.4 + BAI×0.35 + TCI×0.25，相对背景归一化",
         "工业区相对非工业区高强度运行", "0~1"),
        ("", "", "", "", ""),
        ("使用指南", "", "", "", ""),
        ("· CAI > 0.6", "高活跃区域", "与可再生能源申报占比高的企业比对，若CAI高但绿电占比高 → 可疑", "", ""),
        ("· CAI < 0.3", "低活跃区域", "工厂可能处于停工/减产状态，用电申报数据应相应偏低", "", ""),
        ("· CAI 纵向对比", "时间序列趋势", "同一区域不同时段的CAI变化反映实际产能变化", "", ""),
    ]
    for ri, row_vals in enumerate(explanations, 1):
        for ci, val in enumerate(row_vals, 1):
            c = ws3.cell(row=ri, column=ci, value=val)
            if ri == 1:
                c.fill = hdr_fill; c.font = hdr_font
            else:
                c.font = Font(name="Arial", size=10)
            c.border = thin
    for ci, w in enumerate([18, 16, 50, 28, 14], 1):
        ws3.column_dimensions[get_column_letter(ci)].width = w

    wb.save(out_path)
    print(f"\nExcel 报告已保存: {out_path}")


# ── 主流程 ───────────────────────────────────────────────────
def main():
    print(f"Using device: {device}")
    print(f"{'='*60}")
    print("SURF 2026 T2 | Zonal Statistics | 工业区活动指数计算")
    print(f"{'='*60}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vis_dir = os.path.join(OUTPUT_DIR, "vis")
    if SAVE_VIS:
        os.makedirs(vis_dir, exist_ok=True)

    # ── 1. 加载模型 ───────────────────────────────────────────
    with open(CLASSES_JSON) as f:
        class_info = json.load(f)
    num_classes = class_info["num_classes"]
    print(f"类别数: {num_classes}")

    model = load_unet(MODEL_PATH, num_classes, TRAIN_SCRIPT_PATH)

    # ── 2. 发现所有影像文件 ───────────────────────────────────
    img_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    if not img_paths:
        print(f"❌ 未在 {IMAGE_DIR} 找到任何 PNG 文件，请检查路径")
        sys.exit(1)
    print(f"共找到 {len(img_paths)} 张卫星影像 patch")

    # ── 3. 批量推理 ───────────────────────────────────────────
    print(f"\n正在推理（batch_size={BATCH_SIZE}）...")
    inference_results = predict_batch(model, img_paths, INPUT_SIZE)
    print(f"推理完成，共 {len(inference_results)} 张")

    # ── 4. 逐 patch 计算活动指数 ──────────────────────────────
    print("\n正在计算活动指数...")
    records = []
    all_masks   = []   # 用于后续可能的全图拼合
    cai_values  = []
    coverage_values = []

    for idx, (img_path, (pred_mask, rgb)) in enumerate(
            zip(img_paths, inference_results)):

        patch_name = os.path.basename(img_path)
        stats = compute_activity_indices(rgb, pred_mask)

        # 真实标签对比（可选）
        gt_coverage = None
        iou_with_gt  = None
        gt_mask_arr  = None
        if GT_MASK_DIR is not None:
            gt_path = os.path.join(GT_MASK_DIR, patch_name)
            if os.path.exists(gt_path):
                import cv2 as _cv2
                gm = _cv2.imread(gt_path, _cv2.IMREAD_GRAYSCALE)
                if gm is not None:
                    if gm.shape != INPUT_SIZE:
                        gm = _cv2.resize(gm, (INPUT_SIZE[1], INPUT_SIZE[0]),
                                         interpolation=_cv2.INTER_NEAREST)
                    gt_mask_arr = (gm > 0).astype(np.uint8)
                    gt_coverage = round(gt_mask_arr.mean() * 100, 2)
                    # 计算 IoU
                    inter = int(((pred_mask == 1) & (gt_mask_arr == 1)).sum())
                    union = int(((pred_mask == 1) | (gt_mask_arr == 1)).sum())
                    iou_with_gt = round(inter / union, 4) if union > 0 else 0.0

        rec = {
            "patch_name":             patch_name,
            "gt_coverage_pct":        gt_coverage,
            "iou_with_gt":            iou_with_gt,
        }
        rec.update(stats)
        records.append(rec)
        all_masks.append(pred_mask)

        cai = stats.get("composite_activity_index", float("nan"))
        if not np.isnan(cai):
            cai_values.append(cai)
        cov = stats.get("industrial_coverage_pct", 0)
        coverage_values.append(cov)

        # 可视化
        if SAVE_VIS and idx < VIS_MAX_IMAGES:
            vis_path = os.path.join(vis_dir, patch_name)
            save_visualization(rgb, pred_mask, gt_mask_arr, cai, patch_name, vis_path)

        if (idx + 1) % 50 == 0 or idx + 1 == len(img_paths):
            print(f"  [{idx+1}/{len(img_paths)}] "
                  f"CAI={cai:.3f}  工业覆盖={cov:.1f}%")

    # ── 5. 光谱指标（如果有 activity_features.tif）────────────
    spectral_stats = {}
    if ACTIVITY_FEATURES_TIF and os.path.exists(ACTIVITY_FEATURES_TIF):
        print("\n正在从 activity_features.tif 提取光谱指标...")
        # 拼合所有 patch 掩码为一个大图（简化处理：只取均值统计）
        all_masks_stack = np.stack(all_masks)   # (N, H, W)
        combined_mask   = (all_masks_stack.mean(axis=0) > 0.5).astype(np.uint8)
        spectral_stats  = compute_spectral_stats(ACTIVITY_FEATURES_TIF, combined_mask)
        print(f"  光谱指标: {spectral_stats}")

    # ── 6. 汇总 ───────────────────────────────────────────────
    cai_arr = np.array([v for v in cai_values if not np.isnan(v)])
    patches_with_ind = sum(1 for r in records if r.get("industrial_pixel_count", 0) > 0)

    summary = {
        "generated_at":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_patches":          len(records),
        "patches_with_industrial": patches_with_ind,
        "detection_rate":         patches_with_ind / max(len(records), 1),
        "cai_mean":               float(cai_arr.mean())   if len(cai_arr) > 0 else 0,
        "cai_std":                float(cai_arr.std())    if len(cai_arr) > 0 else 0,
        "cai_median":             float(np.median(cai_arr)) if len(cai_arr) > 0 else 0,
        "cai_p75":                float(np.percentile(cai_arr, 75)) if len(cai_arr) > 0 else 0,
        "cai_p95":                float(np.percentile(cai_arr, 95)) if len(cai_arr) > 0 else 0,
        "mean_coverage":          float(np.mean(coverage_values)),
        "max_coverage":           float(np.max(coverage_values)) if coverage_values else 0,
        "total_industrial_pixels": sum(r.get("industrial_pixel_count", 0) for r in records),
    }
    summary.update(spectral_stats)

    # ── 7. 输出 Excel ──────────────────────────────────────────
    excel_path = os.path.join(OUTPUT_DIR, "industrial_activity_index.xlsx")
    write_excel_report(records, summary, excel_path)

    # ── 8. 终端汇总 ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("活动指数汇总")
    print(f"{'='*60}")
    print(f"总 patch 数:        {len(records)}")
    print(f"含工业区 patch 数:  {patches_with_ind} ({patches_with_ind/max(len(records),1):.1%})")
    print(f"CAI 均值:           {summary['cai_mean']:.4f}")
    print(f"CAI 中位数:         {summary['cai_median']:.4f}")
    print(f"CAI 95百分位:       {summary['cai_p95']:.4f}")
    print(f"平均工业覆盖率:     {summary['mean_coverage']:.2f}%")
    print(f"\nExcel → {excel_path}")
    if SAVE_VIS:
        print(f"可视化图 → {vis_dir}/  ({min(VIS_MAX_IMAGES, len(records))} 张)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()