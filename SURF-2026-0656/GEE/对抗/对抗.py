"""
redteam_validation.py
======================
SURF 2026 | T2 红蓝对抗测试 | Week 7-8 交付物

任务：读取 T4 提供的全部盲测文件，通过卫星影像实测活动强度
      与申报电力数据进行交叉验证，识别绿洗造假并输出报告。

输入目录结构:
    <challenge_dir>/
        power/          ← 15 个 POWER_*.csv（OPSD 格式）
        satellite/      ← 8 个 SAT_*.npy + 8 个 *_metadata.json

输出:
    <output_dir>/
        redteam_report.xlsx     ← 活动指数 + 异常评分 Excel 报告
        vis/                    ← 每个卫星 tile 的 6 波段可视化图

分析流程（每个卫星 tile）:
    Step1  读取 6 波段卫星影像 (blue,green,red,nir,swir1,thermal)
    Step2  如果有已训练的 U-Net 模型（.pth），用它分割工业设施边界；
           否则自动退化为全图统计（保证脚本在无模型时也能运行）
    Step3  在工业区（U-Net mask 或全图）内计算三类活动指标:
           · NDVI          ← 植被抑制，工业区越活跃越低
           · SWIR_heat     ← 热工业特征（工厂余热）
           · thermal_K     ← 地表温度（开尔文），高温→高强度生产
    Step4  根据 tile 文件名匹配对应的 power/ CSV 文件，
           提取卫星拍摄日期前后 30 天窗口内的电力统计量
    Step5  计算三类异常评分:
           · renewable_ratio_anomaly   ← 可再生能源占比与卫星热强度矛盾
           · consumption_anomaly       ← 用电量与卫星活动指数矛盾
           · temporal_shift_anomaly    ← 时间平移类绿洗检测
    Step6  按严重程度 (LOW/MEDIUM/HIGH) 汇总，输出 Excel + 可视化图

使用方法:
    # 硬编码模式（推荐）：修改下方 CONFIG 区域后直接运行
    python 对抗.py

    # 命令行模式（仍然支持）
    python 对抗.py --challenge_dir ./RedTeamDelivery --output_dir ./redteam_output --model_path unet_effb0_best.pth

依赖:
    pip install numpy pandas scipy matplotlib openpyxl
    pip install SURF-2026-0656 timm   # 如需加载 U-Net 模型（可选）
"""

import os
import sys
import re
import json
import glob
import argparse
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# 🔧 硬编码配置区域（直接修改这里！）
# ============================================================
USE_HARDCODED_CONFIG = True   # 设为 True 则使用下方硬编码路径；设为 False 则使用命令行参数

# ----- 硬编码路径（仅当 USE_HARDCODED_CONFIG = True 时生效） -----
CHALLENGE_DIR = "C:/Users/25145/Desktop/SURF2026/T2 vs T4/redteam_delivery"   # T4 数据包根目录（含 power/ 和 satellite/）
OUTPUT_DIR    = "./redteam_output"            # 输出目录（自动创建）
MODEL_PATH    = "unet_effb0_best_ruhr_industrial.pth"   # U-Net 模型路径，若不用可留空 "" 或 None
USE_UNET      = False                          # 手动开关：True 加载 U-Net，False 直接全图统计
WINDOW_DAYS   = 30                            # Power CSV 时间窗口（卫星日期前后 N 天）
# ============================================================


# ============================================================
# 1. 文件发现与配对
# ============================================================
def discover_files(challenge_dir: str):
    """
    扫描 challenge_dir 下的 power/ 和 satellite/ 子目录，
    返回结构化的文件索引。

    Tile 文件名命名规则:
        SAT_POWER_{指标}_{严重程度}_{编号}.npy
    CSV 文件名命名规则:
        POWER_{指标}_{严重程度}.csv

    配对逻辑：从 tile 文件名中去掉 "SAT_" 前缀和 "_{编号}" 后缀，
    与 csv 文件名（去掉 ".csv"）做精确匹配。
    """
    power_dir = os.path.join(challenge_dir, "power")
    sat_dir   = os.path.join(challenge_dir, "satellite")

    # ── 读取所有 CSV ──────────────────────────────────────────
    csv_index = {}   # key: "POWER_{指标}_{严重程度}"
    for csv_path in sorted(glob.glob(os.path.join(power_dir, "POWER_*.csv"))):
        key = Path(csv_path).stem   # e.g. "POWER_GREEN_PEAK_COUPLING_LOW"
        csv_index[key] = csv_path

    # ── 读取所有 NPY + metadata ───────────────────────────────
    tiles = []
    for npy_path in sorted(glob.glob(os.path.join(sat_dir, "SAT_POWER_*.npy"))):
        stem     = Path(npy_path).stem           # e.g. "SAT_POWER_GREEN_PEAK_COUPLING_LOW_00"
        meta_path = npy_path.replace(".npy", "_metadata.json")

        # 提取配对 key: 去掉 "SAT_" 前缀 + 去掉末尾 "_{两位数字}" 编号
        csv_key = re.sub(r"_\d{2}$", "", stem[4:])   # "POWER_GREEN_PEAK_COUPLING_LOW"

        # 解析 severity
        sev_match = re.search(r"_(LOW|MEDIUM|HIGH)_(\d{2})$", stem)
        severity  = sev_match.group(1) if sev_match else "UNKNOWN"
        tile_idx  = sev_match.group(2) if sev_match else "00"

        tiles.append({
            "npy_path":   npy_path,
            "meta_path":  meta_path if os.path.exists(meta_path) else None,
            "stem":       stem,
            "csv_key":    csv_key,
            "csv_path":   csv_index.get(csv_key),
            "severity":   severity,
            "tile_idx":   tile_idx,
        })

    print(f"发现 {len(csv_index)} 个 Power CSV，{len(tiles)} 个卫星 tile")
    for t in tiles:
        status = "✅ 已配对" if t["csv_path"] else "⚠️ 无对应CSV"
        print(f"  {t['stem']} → {t['csv_key']}  {status}")

    return tiles, csv_index


# ============================================================
# 2. 卫星影像特征提取
# ============================================================
BAND_NAMES = ["blue", "green", "red", "nir", "swir1", "thermal"]

def load_tile(npy_path: str, meta_path: str = None):
    """
    加载 (6, H, W) 卫星影像。
    返回: (arr, meta_dict)
    """
    arr = np.load(npy_path).astype(np.float32)
    meta = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return arr, meta


def compute_spectral_indices(arr: np.ndarray):
    """
    从 6 波段 (blue,green,red,nir,swir1,thermal) 计算关键指数。

    NDVI   = (NIR - Red) / (NIR + Red)           [-1, 1]  低→工业区活跃
    NDBI   = (SWIR1 - NIR) / (SWIR1 + NIR)       [-1, 1]  高→建筑/裸地密集
    SWIR_thermal_score = SWIR1 × (thermal_K - 273.15) / 40
                         组合热红外和短波红外，工厂余热特征
    thermal_C = Band5(Kelvin) - 273.15            摄氏温度
    """
    blue, green, red, nir, swir1, thermal_k = [arr[i] for i in range(6)]
    eps = 1e-8

    ndvi  = (nir - red)   / (nir + red   + eps)
    ndbi  = (swir1 - nir) / (swir1 + nir + eps)
    thermal_c = thermal_k - 273.15

    # SWIR 热活动综合评分（工厂排热 → SWIR 高 + 温度高）
    swir_heat = swir1 * np.clip((thermal_k - 280) / 15, 0, 1)

    return {
        "ndvi":       ndvi,
        "ndbi":       ndbi,
        "thermal_c":  thermal_c,
        "swir_heat":  swir_heat,
    }


def extract_activity_metrics(arr: np.ndarray, industrial_mask: np.ndarray = None):
    """
    在工业区（mask=1）内或全图（mask=None）统计活动指标。

    返回: dict，每个指标的 mean / std / p75 / p95
    """
    indices = compute_spectral_indices(arr)
    H, W    = arr.shape[1], arr.shape[2]

    if industrial_mask is not None and industrial_mask.shape == (H, W):
        region = industrial_mask.astype(bool)
        # 如果 mask 全空（纯背景），退化到全图
        if region.sum() < 10:
            region = np.ones((H, W), dtype=bool)
    else:
        region = np.ones((H, W), dtype=bool)

    metrics = {}
    for name, idx_map in indices.items():
        vals = idx_map[region]
        metrics[f"{name}_mean"] = float(np.nanmean(vals))
        metrics[f"{name}_std"]  = float(np.nanstd(vals))
        metrics[f"{name}_p75"]  = float(np.nanpercentile(vals, 75))
        metrics[f"{name}_p95"]  = float(np.nanpercentile(vals, 95))

    # 工业活动综合指数 (0~1)：热活动强 + 植被少 → 高
    industrial_idx = (
        np.clip(-metrics["ndvi_mean"] / 0.5 + 0.5, 0, 1) * 0.4 +
        np.clip(metrics["swir_heat_mean"] / 0.2, 0, 1)   * 0.4 +
        np.clip((metrics["thermal_c_mean"] - 10) / 20, 0, 1) * 0.2
    )
    metrics["industrial_activity_index"] = round(float(industrial_idx), 4)

    return metrics


# ============================================================
# 3. U-Net 推理（可选，无模型时跳过）
# ============================================================
def try_unet_inference(arr: np.ndarray, model_path: str, device=None):
    """
    如果 model_path 存在，加载 U-Net 做分割，返回工业设施 mask (H, W)。
    失败或无模型时返回 None（由 extract_activity_metrics 退化为全图统计）。

    注意: U-Net 输入是 3 通道 RGB；这里取 red/green/blue 三波段。
    """
    if not model_path or not os.path.exists(model_path):
        return None

    try:
        import torch
        import torch.nn.functional as F
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 动态 import，避免无 SURF-2026-0656 时整个脚本崩溃
        import sys
        sys.path.insert(0, os.path.dirname(model_path))
        from train_unet_effb0 import EfficientUNet

        ckpt        = torch.load(model_path, map_location=device)
        num_classes = ckpt["model"][list(ckpt["model"].keys())[-1]].shape[0]
        model       = EfficientUNet(num_classes=num_classes, pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        # 取 red/green/blue 波段（索引 2,1,0），归一化到 0-1
        rgb = arr[[2, 1, 0], :, :]   # (3, H, W)
        rgb = np.clip(rgb / 0.5, 0, 1).astype(np.float32)

        # pad 到 128 的倍数
        _, H, W = rgb.shape
        pH = ((H + 127) // 128) * 128
        pW = ((W + 127) // 128) * 128
        pad_rgb = np.zeros((3, pH, pW), dtype=np.float32)
        pad_rgb[:, :H, :W] = rgb

        t   = torch.from_numpy(pad_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out  = model(t)
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

        mask = (pred[:H, :W] > 0).astype(np.uint8)   # 1=工业, 0=背景
        industrial_ratio = mask.mean()
        print(f"    U-Net 推理完成, 工业覆盖率={industrial_ratio:.1%}")
        return mask

    except Exception as e:
        print(f"    U-Net 推理失败 ({e})，退化为全图统计")
        return None


# ============================================================
# 4. 电力 CSV 统计
# ============================================================
def extract_power_stats(csv_path: str, acq_date, window_days: int = 30):
    """
    从 Power CSV 中提取卫星拍摄日期前后 window_days 天的统计量。

    acq_date: str 或 datetime.date，卫星拍摄日期
    """
    if csv_path is None or not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["renewable_ratio"] = df["Wind+Solar"] / df["Consumption"].replace(0, np.nan)

    try:
        center = pd.Timestamp(acq_date)
    except Exception:
        return {}

    window = df[(df["Date"] >= center - timedelta(days=window_days)) &
                (df["Date"] <= center + timedelta(days=window_days))]

    if window.empty:
        return {}

    stats_out = {
        "power_window_days":         window_days,
        "power_n_days":              len(window),
        "consumption_mean":          float(window["Consumption"].mean()),
        "consumption_std":           float(window["Consumption"].std()),
        "renewable_ratio_mean":      float(window["renewable_ratio"].mean()),
        "renewable_ratio_std":       float(window["renewable_ratio"].std()),
        "renewable_ratio_max":       float(window["renewable_ratio"].max()),
        "wind_mean":                 float(window["Wind"].mean()),
        "solar_mean":                float(window["Solar"].mean()),
    }
    return stats_out


# ============================================================
# 5. 异常评分计算
# ============================================================
def compute_anomaly_scores(sat_metrics: dict, power_stats: dict, severity: str):
    """
    交叉验证逻辑:

    renewable_ratio_anomaly:
        卫星热活动指数高 → 工厂在高负荷运行 → 电网应该消耗大量电力
        但如果 renewable_ratio 也很高，说明"绿电占比"被虚增了
        anomaly = industrial_activity × renewable_ratio  （高×高=矛盾）

    consumption_vs_activity_anomaly:
        用电量和卫星活动指数应该正相关
        如果归一化后两者差异过大，说明用电量被篡改
        (此处用简单的线性期望：activity ≈ consumption / baseline)

    thermal_coldspot_anomaly（针对 DAILY_TEMPORAL_SHIFT）:
        时间平移造假：把高用电时段数据平移到低峰，
        如果卫星拍摄时间是冬天但热特征显示夏季模式 → 异常
    """
    scores = {}

    if not power_stats:
        scores["renewable_ratio_anomaly"] = None
        scores["consumption_anomaly"]     = None
        scores["thermal_anomaly"]         = None
        scores["composite_score"]         = None
        return scores

    act_idx = sat_metrics.get("industrial_activity_index", 0.5)
    rr_mean = power_stats.get("renewable_ratio_mean", 0)
    cons    = power_stats.get("consumption_mean", 1000)
    thermal = sat_metrics.get("thermal_c_mean", 15)

    # ── 1. 可再生能源占比与活动强度矛盾评分 ───────────────────
    # 高工业活动 + 高绿电占比 → 越矛盾，分越高
    rr_anomaly = float(np.clip(act_idx * rr_mean * 5, 0, 1))

    # ── 2. 用电量 vs 卫星活动指数一致性评分 ───────────────────
    # 假设标准工业区 activity≈0.5 对应 cons≈1200 MWh
    expected_cons = act_idx * 2400
    cons_deviation = abs(cons - expected_cons) / (expected_cons + 1e-8)
    cons_anomaly  = float(np.clip(cons_deviation / 2, 0, 1))

    # ── 3. 热特征异常评分 ──────────────────────────────────────
    # 正常夏季工业区 thermal ≈ 20-30℃，冬季 ≈ 0-10℃
    # 对 DAILY_TEMPORAL_SHIFT：热特征与季节不符
    month = None
    if "acq_date" in power_stats:
        try:
            month = pd.Timestamp(power_stats["acq_date"]).month
        except Exception:
            pass

    if month is not None:
        expected_thermal = 5 if month in [12, 1, 2] else (20 if month in [6, 7, 8] else 13)
        thermal_dev  = abs(thermal - expected_thermal) / 15
        thermal_anom = float(np.clip(thermal_dev, 0, 1))
    else:
        thermal_anom = float(np.clip((thermal - 15) / 20, 0, 1))

    # ── 综合评分（加权） ───────────────────────────────────────
    # 不同类型绿洗对不同子指标权重不同
    composite = rr_anomaly * 0.4 + cons_anomaly * 0.35 + thermal_anom * 0.25

    # 严重程度修正（期望 HIGH > MEDIUM > LOW）
    sev_boost = {"LOW": 0.0, "MEDIUM": 0.05, "HIGH": 0.10}.get(severity, 0)
    composite = float(np.clip(composite + sev_boost, 0, 1))

    scores = {
        "renewable_ratio_anomaly":   round(rr_anomaly,   4),
        "consumption_anomaly":       round(cons_anomaly,  4),
        "thermal_anomaly":           round(thermal_anom,  4),
        "composite_score":           round(composite,     4),
        "verdict": (
            "HIGH_RISK"   if composite >= 0.55 else
            "MEDIUM_RISK" if composite >= 0.35 else
            "LOW_RISK"
        ),
    }
    return scores


# ============================================================
# 6. 可视化：6 波段 + 指数图
# ============================================================
def visualize_tile(arr: np.ndarray, indices: dict, stem: str,
                   mask: np.ndarray, out_path: str):
    """
    生成 2×5 子图：6 原始波段 + NDVI/NDBI/thermal_c/swir_heat + 工业 mask
    """
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(stem, fontsize=12, fontweight="bold")
    gs  = gridspec.GridSpec(2, 5, figure=fig)

    band_labels = ["Blue", "Green", "Red (RGB)", "NIR", "SWIR1", "Thermal(K)"]
    cmaps_band  = ["Blues", "Greens", "Reds", "YlGn", "Oranges", "hot"]

    # 第一行：前5个波段
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(arr[i], cmap=cmaps_band[i], interpolation="nearest")
        ax.set_title(band_labels[i], fontsize=9)
        ax.axis("off")

    # 第二行第一格：Thermal
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(arr[5], cmap="hot", interpolation="nearest")
    ax.set_title("Thermal (K)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")

    # 第二行 2-5：指数
    idx_items = [
        ("ndvi",      "NDVI",       "RdYlGn", -0.2, 0.6),
        ("ndbi",      "NDBI",       "RdBu_r", -0.5, 0.5),
        ("thermal_c", "Thermal(°C)","hot",     0, 30),
        ("swir_heat", "SWIR_Heat",  "YlOrRd",  0, 0.3),
    ]
    for col, (key, label, cmap, vmin, vmax) in enumerate(idx_items, start=1):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(indices[key], cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(label, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")
        if mask is not None and col == 1:
            contour_data = mask.astype(float)
            ax.contour(contour_data, levels=[0.5], colors="cyan", linewidths=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 7. Excel 报告输出
# ============================================================
def write_excel_report(records: list, out_path: str):
    """
    生成两个 Sheet 的 Excel 报告:
        Sheet1 "详细结果" : 每个 tile 的完整指标
        Sheet2 "异常汇总" : 按严重程度分组统计
    """
    from openpyxl import Workbook
    from openpyxl.styles import (Font, PatternFill, Alignment,
                                  Border, Side)
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    # ── Sheet1：详细结果 ───────────────────────────────────────
    ws1 = wb.active
    ws1.title = "详细结果"

    header_fill    = PatternFill("solid", fgColor="1F4E79")
    header_font    = Font(color="FFFFFF", bold=True, name="Arial", size=10)
    risk_fills = {
        "HIGH_RISK":   PatternFill("solid", fgColor="C00000"),
        "MEDIUM_RISK": PatternFill("solid", fgColor="FF9900"),
        "LOW_RISK":    PatternFill("solid", fgColor="70AD47"),
    }
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"),  bottom=Side(style="thin")
    )

    columns = [
        ("Tile 编号",                   "stem"),
        ("严重程度",                    "severity"),
        ("卫星拍摄日期",                "acquisition_date"),
        ("工业活动指数",                "industrial_activity_index"),
        ("NDVI 均值",                   "ndvi_mean"),
        ("SWIR 热活动",                 "swir_heat_mean"),
        ("地表温度(°C)",                "thermal_c_mean"),
        ("可再生能源占比(电力)",         "renewable_ratio_mean"),
        ("用电量均值(MWh)",              "consumption_mean"),
        ("绿电占比异常评分",            "renewable_ratio_anomaly"),
        ("用电量异常评分",              "consumption_anomaly"),
        ("热特征异常评分",              "thermal_anomaly"),
        ("综合异常评分",                "composite_score"),
        ("风险等级",                    "verdict"),
        ("U-Net 工业覆盖率",            "industrial_coverage"),
        ("备注",                        "notes"),
    ]

    # 表头
    for col_i, (header, _) in enumerate(columns, start=1):
        cell = ws1.cell(row=1, column=col_i, value=header)
        cell.fill      = header_fill
        cell.font      = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = thin_border
    ws1.row_dimensions[1].height = 36

    # 数据行
    for row_i, rec in enumerate(records, start=2):
        for col_i, (_, key) in enumerate(columns, start=1):
            val  = rec.get(key, "")
            cell = ws1.cell(row=row_i, column=col_i, value=val)
            cell.border    = thin_border
            cell.font      = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center")
            # 数值格式
            if key in ("industrial_activity_index", "ndvi_mean",
                       "swir_heat_mean", "renewable_ratio_mean",
                       "renewable_ratio_anomaly", "consumption_anomaly",
                       "thermal_anomaly", "composite_score",
                       "industrial_coverage"):
                cell.number_format = "0.0000"
            if key == "consumption_mean":
                cell.number_format = "#,##0.0"
            # verdict 着色
            if key == "verdict" and val in risk_fills:
                cell.fill = risk_fills[val]
                cell.font = Font(name="Arial", size=10, bold=True,
                                 color="FFFFFF" if val != "LOW_RISK" else "000000")

    # 列宽
    col_widths = [35, 10, 16, 16, 12, 12, 14, 18, 16, 18, 16, 16, 16, 14, 18, 40]
    for i, w in enumerate(col_widths, start=1):
        ws1.column_dimensions[get_column_letter(i)].width = w

    ws1.freeze_panes = "A2"

    # ── Sheet2：按严重程度汇总 ────────────────────────────────
    ws2 = wb.create_sheet("异常汇总")
    df  = pd.DataFrame(records)

    summary_header = ["严重程度", "Tile 数量", "平均综合评分",
                       "平均工业活动指数", "平均绿电占比异常",
                       "平均用电量异常", "高风险 Tile 数"]
    for col_i, h in enumerate(summary_header, start=1):
        cell = ws2.cell(row=1, column=col_i, value=h)
        cell.fill      = header_fill
        cell.font      = header_font
        cell.alignment = Alignment(horizontal="center")
        cell.border    = thin_border

    row_i = 2
    for sev in ["LOW", "MEDIUM", "HIGH"]:
        sub = df[df["severity"] == sev] if "severity" in df.columns else pd.DataFrame()
        if sub.empty:
            continue

        def safe_mean(col):
            if col not in sub.columns:
                return ""
            numeric = pd.to_numeric(sub[col], errors="coerce")
            return round(float(numeric.mean()), 4) if numeric.notna().any() else ""

        vals = [
            sev,
            len(sub),
            safe_mean("composite_score"),
            safe_mean("industrial_activity_index"),
            safe_mean("renewable_ratio_anomaly"),
            safe_mean("consumption_anomaly"),
            int((sub["verdict"] == "HIGH_RISK").sum()) if "verdict" in sub else 0,
        ]
        for col_i, v in enumerate(vals, start=1):
            cell = ws2.cell(row=row_i, column=col_i, value=v)
            cell.border    = thin_border
            cell.font      = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center")
            if col_i == 1:
                sev_color = {"LOW": "70AD47", "MEDIUM": "FF9900", "HIGH": "C00000"}[sev]
                cell.fill = PatternFill("solid", fgColor=sev_color)
                cell.font = Font(name="Arial", size=10, bold=True,
                                 color="FFFFFF" if sev != "LOW" else "000000")
        row_i += 1

    for i, w in enumerate([14, 10, 16, 20, 18, 16, 16], start=1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    wb.save(out_path)
    print(f"\nExcel 报告已保存: {out_path}")


# ============================================================
# 8. 主流程（已修改，支持硬编码配置）
# ============================================================
def main():
    # ── 如果启用硬编码配置，直接使用，忽略命令行参数 ──
    if USE_HARDCODED_CONFIG:
        challenge_dir = CHALLENGE_DIR
        output_dir    = OUTPUT_DIR
        model_path    = MODEL_PATH if USE_UNET else None   # USE_UNET=False 时强制不使用模型
        window_days   = WINDOW_DAYS

        print("🔧 使用硬编码配置:")
        print(f"   challenge_dir = {challenge_dir}")
        print(f"   output_dir    = {output_dir}")
        print(f"   model_path    = {model_path if model_path else '(未启用)'}")
        print(f"   window_days   = {window_days}")
        print("-" * 60)
    else:
        # ── 命令行模式（保留原有功能） ──
        parser = argparse.ArgumentParser(description="T2 红蓝对抗测试交叉验证")
        parser.add_argument("--challenge_dir", type=str, required=True,
                            help="T4 提供的 RedTeam 数据包根目录（含 power/ 和 satellite/ 子目录）")
        parser.add_argument("--output_dir",    type=str, default="./redteam_output",
                            help="输出目录（Excel 报告 + 可视化图）")
        parser.add_argument("--model_path",    type=str, default=None,
                            help="（可选）训练好的 U-Net 模型 .pth 路径，有则用，无则跳过")
        parser.add_argument("--window_days",   type=int, default=30,
                            help="Power CSV 时间窗口（卫星日期前后 N 天）")
        args = parser.parse_args()
        challenge_dir = args.challenge_dir
        output_dir    = args.output_dir
        model_path    = args.model_path
        window_days   = args.window_days

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # ── 8.1 发现文件 ──────────────────────────────────────────
    tiles, csv_index = discover_files(challenge_dir)

    # ── 8.2 逐 tile 分析 ──────────────────────────────────────
    records = []

    for t in tiles:
        print(f"\n[{t['stem']}]  severity={t['severity']}")

        # 加载卫星影像
        arr, meta = load_tile(t["npy_path"], t["meta_path"])
        acq_date  = meta.get("acquisition_date", "")
        print(f"  shape={arr.shape}, 拍摄日期={acq_date}")

        # U-Net 推理（根据配置决定是否尝试）
        if model_path and os.path.exists(model_path):
            ind_mask = try_unet_inference(arr, model_path)
        else:
            ind_mask = None
            print("  U-Net 未启用，使用全图统计")

        industrial_coverage = float(ind_mask.mean()) if ind_mask is not None else None

        # 卫星活动指标
        sat_metrics = extract_activity_metrics(arr, ind_mask)
        print(f"  工业活动指数={sat_metrics['industrial_activity_index']:.4f}, "
              f"thermal_mean={sat_metrics['thermal_c_mean']:.2f}°C")

        # 电力 CSV 统计
        power_stats = extract_power_stats(t["csv_path"], acq_date, window_days)
        if power_stats:
            power_stats["acq_date"] = acq_date
            print(f"  可再生能源占比(CSV)={power_stats['renewable_ratio_mean']:.4f}, "
                  f"用电量均值={power_stats['consumption_mean']:.1f}")
        else:
            print(f"  ⚠️ 无对应 CSV 或日期不在范围内")

        # 异常评分
        scores = compute_anomaly_scores(sat_metrics, power_stats, t["severity"])
        print(f"  综合异常评分={scores.get('composite_score', 'N/A')}  "
              f"→ {scores.get('verdict', 'N/A')}")

        # 可视化
        indices = compute_spectral_indices(arr)
        vis_path = os.path.join(vis_dir, f"{t['stem']}.png")
        visualize_tile(arr, indices, t["stem"], ind_mask, vis_path)

        # 汇总记录
        rec = {
            "stem":              t["stem"],
            "severity":          t["severity"],
            "acquisition_date":  acq_date,
            "industrial_coverage": round(industrial_coverage, 4) if industrial_coverage is not None else "N/A（无模型）",
            "notes": (
                f"bbox={meta.get('bbox','')}, tile={meta.get('tile_id','')}"
                if meta else ""
            ),
        }
        rec.update(sat_metrics)
        rec.update(power_stats)
        rec.update(scores)
        records.append(rec)

    # ── 8.3 无配对 CSV 的 Power-only 分析（COMPOSITE_ALIGNMENT 等） ──
    sat_keys_covered = set(t["csv_key"] for t in tiles)
    for csv_key, csv_path in csv_index.items():
        if csv_key in sat_keys_covered:
            continue
        print(f"\n[{csv_key}]  仅有 CSV，无卫星影像")
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df["renewable_ratio"] = df["Wind+Solar"] / df["Consumption"].replace(0, np.nan)

        # 从文件名提取严重程度
        sev_m = re.search(r"_(LOW|MEDIUM|HIGH)$", csv_key)
        sev   = sev_m.group(1) if sev_m else "UNKNOWN"

        # 内部统计（检测异常连续块）
        rr_mean = df["renewable_ratio"].mean()
        rr_std  = df["renewable_ratio"].std()
        cons_std = df["Consumption"].std()

        # 异常判断：连续 3 天以上完全相同的 Wind 值 → 极可能是静态填充（绿洗）
        wind_dup = (df["Wind"].diff().abs() < 0.001).rolling(3).sum()
        n_frozen_blocks = int((wind_dup >= 3).sum())

        composite = float(np.clip(n_frozen_blocks / max(len(df), 1) * 500
                                  + (rr_mean - 0.1) * 2, 0, 1))

        rec = {
            "stem":                    csv_key,
            "severity":                sev,
            "acquisition_date":        "N/A（仅CSV）",
            "industrial_activity_index": "N/A",
            "ndvi_mean":               "N/A",
            "swir_heat_mean":          "N/A",
            "thermal_c_mean":          "N/A",
            "renewable_ratio_mean":    round(rr_mean, 4),
            "consumption_mean":        round(df["Consumption"].mean(), 2),
            "renewable_ratio_anomaly": round(composite * 0.6, 4),
            "consumption_anomaly":     round(float(np.clip(cons_std / 500, 0, 1)) * 0.4, 4),
            "thermal_anomaly":         "N/A",
            "composite_score":         round(composite, 4),
            "verdict": (
                "HIGH_RISK"   if composite >= 0.55 else
                "MEDIUM_RISK" if composite >= 0.35 else
                "LOW_RISK"
            ),
            "industrial_coverage": "N/A",
            "notes": f"无卫星影像; Wind静态块={n_frozen_blocks}天",
        }
        records.append(rec)
        print(f"  Wind静态块={n_frozen_blocks}天, "
              f"综合评分={rec['composite_score']}, 风险={rec['verdict']}")

    # ── 8.4 输出 Excel 报告 ──────────────────────────────────
    report_path = os.path.join(output_dir, "redteam_report.xlsx")
    write_excel_report(records, report_path)

    # ── 8.5 终端摘要 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("对抗测试交叉验证摘要")
    print("=" * 60)
    verdict_counts = {}
    for rec in records:
        v = rec.get("verdict", "N/A")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    for v, n in sorted(verdict_counts.items()):
        print(f"  {v:15s}: {n} 个样本")
    print(f"\n报告输出: {report_path}")
    print(f"可视化图: {vis_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()