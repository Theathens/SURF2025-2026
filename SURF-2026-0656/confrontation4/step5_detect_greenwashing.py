# step5_detect_greenwashing.py
"""
Step 5: 规则库交叉验证（升级版 — 集成 structured_claims.json）

输入:
  - facility_trends.csv  (step3)
  - facility_panel.csv   (step3)
  - company_trends.csv   (step4)
  - unified_annual_metrics.csv (step1)
  - structured_claims.json (LLM 提取的声明清单)
  - 各 cohort 的 ESG 报告 md (可选, 用于文本层匹配)

输出:
  - raw_findings.csv     统一的原始发现表, 每行含 detection_channel ∈
                         {satellite, structured, text}, 具备三张模板所需的所有列
"""
import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Optional

from config import (
    FACILITY_TRENDS_OUTPUT, COMPANY_TRENDS_OUTPUT,
    UNIFIED_METRICS_OUTPUT, FINDINGS_OUTPUT, DATA_DIR,
    VIRTUAL_REPORTS_DIR, REAL_REPORTS_DIR,
    THRESHOLDS, CONFIDENCE_BASE, DEFAULT_CONFIDENCE,
    COHORT_VIRTUAL, COHORT_REAL, TEXT_DETECTION_CONFIG,
)

FACILITY_PANEL = os.path.join(DATA_DIR, "facility_panel.csv")
YOY_DETAILS_FILE = os.path.join(DATA_DIR, "company_yoy_details.csv")
STRUCTURED_CLAIMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(FACILITY_TRENDS_OUTPUT)),  # 回到 confrontation4/
    "..", "confrontation3", "26.7.26", "05_group3_public_blind_20260724",
    "structured_claims.json"
)
# 如果上述路径不存在，尝试从 PACKAGE_DIR 读取
if not os.path.exists(STRUCTURED_CLAIMS_PATH):
    from config import PACKAGE_DIR
    STRUCTURED_CLAIMS_PATH = os.path.join(PACKAGE_DIR, "structured_claims.json")


# ------------------ 工具 ------------------
def clip_conf(x: float) -> float:
    return max(0.0, min(0.99, round(x, 2)))


def parse_area_from_text(text: str):
    """
    从中文/英文文本中提取面积: '万平方米 / m² / km² / hectare'
    返回 (m², 原文片段) 或 (None, None)
    """
    if not isinstance(text, str):
        return None, None
    patterns = [
        (re.compile(r"(\d+\.?\d*)\s*万\s*(平方米|㎡|m²|m\^2)"), lambda v: v * 10000),
        (re.compile(r"(\d+\.?\d*)\s*(平方米|㎡|m²|m\^2)"), lambda v: v),
        (re.compile(r"(\d+\.?\d*)\s*(公顷|hectare|ha)\b", re.IGNORECASE), lambda v: v * 10000),
        (re.compile(r"(\d+\.?\d*)\s*(km²|平方公里|sq\s*km)", re.IGNORECASE), lambda v: v * 1_000_000),
    ]
    for pat, conv in patterns:
        m = pat.search(text)
        if m:
            v = float(m.group(1))
            return conv(v), m.group(0)
    return None, None


def parse_capacity_mw(text: str):
    if not isinstance(text, str):
        return None, None
    m = re.search(r"(\d+\.?\d*)\s*(MW|兆瓦|megawatt|MWp)", text, re.IGNORECASE)
    if m:
        return float(m.group(1)), m.group(0)
    m = re.search(r"(\d+\.?\d*)\s*(kW|千瓦)", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 1000.0, m.group(0)
    return None, None


def parse_energy_mwh(text: str):
    """从文本中提取能源消耗量（MWh）。"""
    if not isinstance(text, str):
        return None, None
    # 匹配 MWh 或 GWh
    m = re.search(r"(\d+[\d,]*\.?\d*)\s*(MWh|兆瓦时)", text, re.IGNORECASE)
    if m:
        val_str = m.group(1).replace(",", "")
        return float(val_str), m.group(0)
    m = re.search(r"(\d+[\d,]*\.?\d*)\s*(GWh|吉瓦时)", text, re.IGNORECASE)
    if m:
        val_str = m.group(1).replace(",", "")
        return float(val_str) * 1000.0, m.group(0)
    return None, None


def parse_emissions_tco2e(text: str):
    """从文本中提取排放量（tCO2e）。"""
    if not isinstance(text, str):
        return None, None
    m = re.search(r"(\d+[\d,]*\.?\d*)\s*(tCO2e|吨二氧化碳当量|万吨CO2)", text, re.IGNORECASE)
    if m:
        val_str = m.group(1).replace(",", "")
        multiplier = 10000.0 if "万吨" in m.group(2) else 1.0
        return float(val_str) * multiplier, m.group(0)
    return None, None


def load_reports(cohort: str) -> dict:
    """返回 {firm_id_prefix: full_text}, 后续 startswith 匹配。"""
    if cohort == COHORT_VIRTUAL:
        d = VIRTUAL_REPORTS_DIR
    else:
        d = REAL_REPORTS_DIR
    out = {}
    if not os.path.isdir(d):
        return out
    for fname in os.listdir(d):
        if fname.endswith(".md"):
            path = os.path.join(d, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out[fname] = f.read()
            except Exception:
                pass
    return out


def find_report_for_firm(reports: dict, firm_id: str) -> Optional[str]:
    """按 firm_id 或其片段在文件名中查找。"""
    firm_lc = firm_id.lower()
    for fname, txt in reports.items():
        stem = fname.lower()
        if firm_lc in stem or firm_lc.split("_")[0] in stem:
            return fname
    return None


# ------------------ 单元记录构造 ------------------
def resolve_claim_facility(claim: dict, facility_df: pd.DataFrame, year_col: str = None) -> Optional[pd.Series]:
    """structured_claims.json 无 facility_id 字段，把声称解析到具体设施行。

    year_col 为 None 时按 (firm_id) 匹配（设施趋势表，每设施一行）；
    否则按 (firm_id, year) 匹配（设施面板表）。
    多设施公司先用 claim_text / report_file / section 中的设施名可区分片段匹配，
    失败则取第一条（一致性比对仍有效）。
    """
    firm_id = claim["firm_id"]
    rows = facility_df[facility_df["firm_id"] == firm_id]
    if year_col is not None:
        rows = rows[rows[year_col] == claim.get("year")]
    if rows.empty:
        return None
    if len(rows) == 1:
        return rows.iloc[0]
    text = " ".join(str(claim.get(k) or "") for k in ("claim_text", "report_file", "section")).lower()
    for _, r in rows.iterrows():
        fname = str(r.get("facility_name", "")).lower()
        comp = str(r.get("company_name", "")).lower()
        frag = fname.replace(comp, "").strip()
        if frag and frag in text:
            return r
    return rows.iloc[0]


def _base(record: dict, fac: pd.Series, comp: pd.Series) -> dict:
    record.update({
        "cohort": fac["cohort"],
        "firm_id": fac["firm_id"],
        "company_name": fac.get("company_name") or comp.get("company_name"),
        "facility_id": fac["facility_id"],
        "facility_name": fac.get("facility_name"),
    })
    return record


# ============================================================
# 原有规则（保持不变）
# ============================================================
def rule_expansion_vs_energy(fac, comp) -> List[dict]:
    """建成区显著扩张但公司能耗未同步增长。"""
    out = []
    built_pct = fac.get("built_pct_change")
    energy_pct = comp.get("energy_pct_change")
    if pd.isna(built_pct) or pd.isna(energy_pct):
        return out
    if built_pct > THRESHOLDS["built_expansion_pct"] and energy_pct < THRESHOLDS["energy_growth_pct"]:
        conf = CONFIDENCE_BASE["physical_expansion_energy_mismatch"]
        conf += min(0.15, (built_pct - THRESHOLDS["built_expansion_pct"]) / 100.0)
        conf -= min(0.10, max(0, energy_pct) / 100.0)
        rec = _base({
            "detection_channel": "satellite",
            "aoi_id": fac.get("aoi_id_end"),
            "year": int(fac["end_year"]),
            "image_date": fac.get("image_date_end"),
            "variable": "built_up_area_ha",
            "reported_value": fac.get("built_end_ha"),
            "recalculated_value": fac.get("built_end_ha"),
            "unit": "ha",
            "spatial_boundary": fac.get("aoi_bbox") or fac.get("spatial_boundary_hint") or "AOI bbox",
            "data_source": "step3 facility_trends; unified_annual_metrics",
            "calculation": (f"AOI built-up 由 {fac.get('built_start_ha'):.2f}ha "
                            f"({fac['start_year']}) 增至 {fac.get('built_end_ha'):.2f}ha "
                            f"({fac['end_year']}), +{built_pct:.1f}%; 公司总能耗同期变化 {energy_pct:.1f}%."),
            "confidence": clip_conf(conf),
            "conclusion": "physical_expansion_energy_mismatch",
            "notes": "厂区物理扩张与公司披露能耗增长不匹配, 疑似漏报能耗或产量转移.",
        }, fac, comp)
        out.append(rec)
    return out


def rule_pv_growth_vs_renewable(fac, comp) -> List[dict]:
    out = []
    pv_pct = fac.get("pv_pct_change")
    renew_delta = comp.get("renewable_ratio_delta_pp")
    if pd.isna(pv_pct) or pd.isna(renew_delta):
        return out
    if pv_pct > THRESHOLDS["pv_growth_pct"] and renew_delta < THRESHOLDS["renewable_delta_pp"]:
        conf = CONFIDENCE_BASE["pv_growth_renewable_mismatch"]
        conf += min(0.15, (pv_pct - THRESHOLDS["pv_growth_pct"]) / 200.0)
        rec = _base({
            "detection_channel": "satellite",
            "aoi_id": fac.get("aoi_id_end"),
            "year": int(fac["end_year"]),
            "image_date": fac.get("image_date_end"),
            "variable": "roof_pv_area_m2",
            "reported_value": fac.get("pv_end_m2"),
            "recalculated_value": fac.get("pv_end_m2"),
            "unit": "m2",
            "spatial_boundary": fac.get("aoi_bbox") or "AOI bbox",
            "data_source": "step3 facility_trends; unified_annual_metrics",
            "calculation": (f"屋顶 PV 面积由 {fac.get('pv_start_m2'):.0f} m² 增至 "
                            f"{fac.get('pv_end_m2'):.0f} m² (+{pv_pct:.1f}%); "
                            f"公司可再生电力占比仅提升 {renew_delta:.2f} 个百分点."),
            "confidence": clip_conf(conf),
            "conclusion": "pv_growth_renewable_mismatch",
            "notes": "屋顶光伏面积显著增加, 但披露的可再生电力占比未同步上升.",
        }, fac, comp)
        out.append(rec)
    return out


def rule_nightlight_vs_production(fac, comp) -> List[dict]:
    out = []
    nl_delta = fac.get("nightlight_delta")
    prod_pct = comp.get("prod_pct_change")
    if pd.isna(nl_delta) or pd.isna(prod_pct):
        return out
    if nl_delta <= THRESHOLDS["nightlight_delta_min"] and prod_pct > THRESHOLDS["production_growth_pct"]:
        conf = CONFIDENCE_BASE["nightlight_production_mismatch"]
        conf += min(0.10, prod_pct / 400.0)
        rec = _base({
            "detection_channel": "satellite",
            "aoi_id": fac.get("aoi_id_end"),
            "year": int(fac["end_year"]),
            "image_date": fac.get("image_date_end"),
            "variable": "annual_night_light_mean",
            "reported_value": fac.get("nightlight_end"),
            "recalculated_value": fac.get("nightlight_end"),
            "unit": "nW/cm2/sr",
            "spatial_boundary": fac.get("aoi_bbox") or "AOI bbox",
            "data_source": "VIIRS DNB monthly composite; unified_annual_metrics",
            "calculation": (f"夜光强度年均变化 Δ={nl_delta:.2f} (无明显上升); "
                            f"公司产量同期 +{prod_pct:.1f}%."),
            "confidence": clip_conf(conf),
            "conclusion": "nightlight_production_mismatch",
            "notes": "产量声称大幅上升但夜光强度未上升, 疑似产量口径夸大或存在外部代工.",
        }, fac, comp)
        out.append(rec)
    return out


def rule_pv_capacity_area_consistency(fac, comp) -> List[dict]:
    """
    PV 面积 → 名义装机的自洽性 (15000 m²/MW)。
    已禁用：该规则在真实公司数据中无法使用。
    """
    return []  # 禁用


def rule_heat_vs_scope1(fac, comp) -> List[dict]:
    out = []
    heat_delta = fac.get("heat_delta")
    scope1_pct = comp.get("scope1_pct_change")
    if pd.isna(heat_delta) or pd.isna(scope1_pct):
        return out
    if heat_delta > THRESHOLDS["heat_index_delta_min"] and scope1_pct <= THRESHOLDS["scope1_growth_pct"]:
        conf = CONFIDENCE_BASE["heat_scope1_mismatch"] + min(0.15, heat_delta / 5.0)
        rec = _base({
            "detection_channel": "satellite",
            "aoi_id": fac.get("aoi_id_end"),
            "year": int(fac["end_year"]),
            "image_date": fac.get("image_date_end"),
            "variable": "land_surface_heat_index",
            "reported_value": fac.get("heat_end"),
            "recalculated_value": fac.get("heat_end"),
            "unit": "°C (AOI - buffer ring)",
            "spatial_boundary": fac.get("aoi_bbox") or "AOI bbox vs 3km ring",
            "data_source": "MODIS/061/MOD11A2; unified_annual_metrics",
            "calculation": (f"AOI 热岛指数由 {fac.get('heat_start'):.2f} 升至 {fac.get('heat_end'):.2f} "
                            f"(Δ={heat_delta:.2f}°C); 公司 Scope1 变化 {scope1_pct:.1f}%."),
            "confidence": clip_conf(conf),
            "conclusion": "heat_scope1_mismatch",
            "notes": "厂区热岛显著上升暗示直接排放活动增强, 但披露 Scope1 未上升.",
        }, fac, comp)
        out.append(rec)
    return out


def rule_energy_intensity_drop(comp) -> List[dict]:
    """公司能耗强度骤降，作为结构化质疑发现（非遥感）。"""
    out = []
    ei = comp.get("energy_intensity_pct_change")
    if pd.isna(ei):
        return out
    if ei < THRESHOLDS["energy_intensity_drop_pct"]:
        conf = CONFIDENCE_BASE["energy_intensity_suspicious_drop"] + min(0.15, abs(ei) / 100.0)
        out.append({
            "detection_channel": "structured",
            "cohort": comp["cohort"],
            "firm_id": comp["firm_id"],
            "company_name": comp.get("company_name"),
            "facility_id": None,
            "facility_name": None,
            "aoi_id": None,
            "year": comp.get("energy_intensity_end_year"),
            "image_date": None,
            "variable": "energy_intensity_mwh_per_production_unit",
            "reported_value": comp.get("energy_intensity_end"),
            "recalculated_value": comp.get("energy_intensity_end"),
            "unit": "MWh/production_unit",
            "spatial_boundary": None,
            "data_source": "annual_esg_metrics",
            "calculation": (f"能耗强度由 {comp.get('energy_intensity_start'):.4f} "
                            f"降至 {comp.get('energy_intensity_end'):.4f} ({ei:.1f}%)."),
            "confidence": clip_conf(conf),
            "conclusion": "energy_intensity_suspicious_drop",
            "notes": "能耗强度大幅下降, 需要遥感或工艺证据支撑其真实性.",
        })
    return out


def rule_report_pv_claim_vs_remote(fac, reports) -> List[dict]:
    """在企业 ESG 报告中查找与该 facility 关联的 PV 面积/装机声明并与遥感值比较。"""
    out = []
    fname = find_report_for_firm(reports, fac["firm_id"])
    if fname is None:
        return out
    text = reports[fname]
    name_lc = str(fac.get("facility_name", "")).lower()
    hits = []
    for para in re.split(r"\n\s*\n", text):
        p_lc = para.lower()
        if name_lc and any(tok in p_lc for tok in [name_lc, name_lc.split()[0] if name_lc else ""]):
            hits.append(para)
    if not hits:
        hits = [p for p in re.split(r"\n\s*\n", text) if "光伏" in p or "PV" in p.upper()]
    end_pv = fac.get("pv_end_m2")
    end_cap = fac.get("pv_capacity_end_mw")
    for para in hits:
        area_m2, area_str = parse_area_from_text(para)
        cap_mw, cap_str = parse_capacity_mw(para)
        if area_m2 is not None and pd.notna(end_pv) and end_pv > 0:
            dev = (area_m2 - end_pv) / end_pv * 100.0
            if abs(dev) > THRESHOLDS["pv_claim_dev_pct"]:
                conf = CONFIDENCE_BASE["claim_pv_inflation"] + min(0.15, abs(dev) / 200.0)
                out.append(_base({
                    "detection_channel": "text",
                    "aoi_id": fac.get("aoi_id_end"),
                    "year": int(fac["end_year"]),
                    "image_date": fac.get("image_date_end"),
                    "variable": "roof_pv_area (text_claim vs remote)",
                    "reported_value": area_m2,
                    "recalculated_value": end_pv,
                    "unit": "m2",
                    "spatial_boundary": fac.get("aoi_bbox") or "AOI bbox",
                    "data_source": f"ESG report: {fname}",
                    "calculation": (f"报告声明 PV 面积约 {area_str} ({area_m2:.0f} m²); "
                                    f"遥感 {fac['end_year']} 观测 {end_pv:.0f} m², 偏差 {dev:.1f}%."),
                    "confidence": clip_conf(conf),
                    "conclusion": "claim_pv_inflation" if dev > 0 else "claim_pv_conservative",
                    "notes": para.strip()[:300],
                    "report_file": fname,
                }, fac, {"company_name": fac.get("company_name")}))
        if cap_mw is not None and pd.notna(end_cap) and end_cap > 0:
            dev = (cap_mw - end_cap) / end_cap * 100.0
            if abs(dev) > THRESHOLDS["pv_capacity_dev_pct"]:
                conf = CONFIDENCE_BASE["pv_capacity_area_mismatch"] + min(0.15, abs(dev) / 200.0)
                out.append(_base({
                    "detection_channel": "text",
                    "aoi_id": fac.get("aoi_id_end"),
                    "year": int(fac["end_year"]),
                    "image_date": fac.get("image_date_end"),
                    "variable": "estimated_rooftop_pv_capacity_mw (text vs remote)",
                    "reported_value": cap_mw,
                    "recalculated_value": end_cap,
                    "unit": "MW",
                    "spatial_boundary": fac.get("aoi_bbox") or "AOI bbox",
                    "data_source": f"ESG report: {fname}",
                    "calculation": (f"报告声明装机 {cap_str} ({cap_mw:.2f} MW); "
                                    f"遥感面积推算 {end_cap:.2f} MW, 偏差 {dev:.1f}%."),
                    "confidence": clip_conf(conf),
                    "conclusion": "pv_capacity_area_mismatch",
                    "notes": para.strip()[:300],
                    "report_file": fname,
                }, fac, {"company_name": fac.get("company_name")}))
    return out


# ============================================================
# 规则：过程措辞扫描
# ============================================================
def rule_process_wording(reports: dict, firm_id: str, cohort: str) -> List[dict]:
    """
    扫描报告中的过程措辞（draft/placeholder/TBD/内部/待定等）。
    """
    out = []
    fname = find_report_for_firm(reports, firm_id)
    if fname is None:
        return out
    text = reports[fname]
    keywords = TEXT_DETECTION_CONFIG["process_wording_keywords"]

    for i, line in enumerate(text.split('\n'), 1):
        line_lower = line.lower()
        for kw in keywords:
            if kw.lower() in line_lower and len(line.strip()) > 10:
                out.append({
                    "detection_channel": "text",
                    "cohort": cohort,
                    "firm_id": firm_id,
                    "company_name": None,
                    "facility_id": None,
                    "facility_name": None,
                    "aoi_id": None,
                    "year": None,
                    "image_date": None,
                    "variable": "process_wording",
                    "reported_value": None,
                    "recalculated_value": None,
                    "unit": None,
                    "spatial_boundary": None,
                    "data_source": f"ESG report: {fname}",
                    "calculation": f"Line {i} 含过程措辞关键词 '{kw}': {line.strip()[:150]}",
                    "confidence": clip_conf(CONFIDENCE_BASE["process_wording_detected"]),
                    "conclusion": "process_wording_detected",
                    "notes": f"该行包含未清理的内部/测试措辞，应在正式披露前移除。",
                    "report_file": fname,
                })
                break
    return out


# ============================================================
# 规则：YoY 异常检测（基于 company_yoy_details.csv）
# ============================================================
def rule_yoy_anomaly_from_details(yoy_df: pd.DataFrame) -> List[dict]:
    """
    year-over-year 异常检测使用长表 company_yoy_details.csv。
    遍历所有 (firm, metric, year) YoY 记录，对超阈值的每一年都单独上报。
    """
    out = []

    thresholds = {
        "electricity_mwh":                    {"upper": 30.0, "lower": -20.0, "conf": 0.70, "conclusion": "yoy_anomaly_electricity", "note": "单年电耗变化异常，疑似中间年数据注水或口径突变。"},
        "total_energy_mwh":                   {"upper": 30.0, "lower": -20.0, "conf": 0.70, "conclusion": "yoy_anomaly_energy", "note": "单年总能耗变化异常，疑似数据篡改。"},
        "renewable_electricity_ratio_pct":    {"upper": 10.0, "lower": -10.0, "conf": 0.75, "conclusion": "yoy_anomaly_renewable_ratio", "note": "可再生电力占比单年变化超过 10 个百分点，不符合能源转型的渐进性特征。"},
        "scope1_emission_tco2e":              {"upper": 40.0, "lower": -30.0, "conf": 0.65, "conclusion": "yoy_anomaly_scope1", "note": "Scope 1 排放单年变化异常。"},
        "scope2_market_based_tco2e":          {"upper": 40.0, "lower": -30.0, "conf": 0.65, "conclusion": "yoy_anomaly_scope2", "note": "Scope 2 排放单年变化异常。"},
        "scope3_emission_tco2e":              {"upper": 40.0, "lower": -30.0, "conf": 0.60, "conclusion": "yoy_anomaly_scope3", "note": "Scope 3 排放单年变化异常。"},
        "green_power_purchased_mwh":          {"upper": 100.0, "lower": -50.0, "conf": 0.60, "conclusion": "yoy_anomaly_green_purchased", "note": "绿电采购量单年变化异常。"},
    }

    for _, r in yoy_df.iterrows():
        if r["cohort"] == COHORT_VIRTUAL:
            continue
        metric = r["metric"]
        if metric not in thresholds:
            continue
        cfg = thresholds[metric]
        signed = r["signed_yoy"]
        if pd.isna(signed):
            continue
        if signed > cfg["upper"] or signed < cfg["lower"]:
            unit_label = r.get("unit", "%")
            out.append({
                "detection_channel": "structured",
                "cohort": r["cohort"],
                "firm_id": r["firm_id"],
                "company_name": r.get("company_name"),
                "facility_id": None,
                "facility_name": None,
                "aoi_id": None,
                "year": int(r["curr_year"]),
                "image_date": None,
                "variable": metric,
                "reported_value": r["curr_value"],
                "recalculated_value": r["prev_value"],
                "unit": unit_label,
                "spatial_boundary": None,
                "data_source": "annual_esg_metrics year-over-year analysis",
                "calculation": (f"{int(r['prev_year'])}→{int(r['curr_year'])}: {metric} 由 "
                                f"{r['prev_value']:,.4g} 变为 {r['curr_value']:,.4g}, "
                                f"单年变化 {signed:+.2f}{unit_label} "
                                f"(阈值区间 [{cfg['lower']:+.1f}, {cfg['upper']:+.1f}]{unit_label})"),
                "confidence": cfg["conf"],
                "conclusion": cfg["conclusion"],
                "notes": cfg["note"],
            })
    return out


def rule_emission_electricity_decoupling(metrics_df: pd.DataFrame) -> List[dict]:
    """
    规则：排放与电量脱钩检测（识别"排放轨迹平滑"型造假）。

    物理前提：直接排放（Scope 1）随活动水平同向变动。若电量大幅变化而排放
    反向大幅变化，排放轨迹大概率被人为平滑/改写。
    判据：|Δ排放| >= EMIS_MIN 且 |Δ电量| >= ELEC_MIN 且二者方向相反。
    实例：LONGi 2022→2023 电量 +35.3% 而 Scope 1 -14.1%（6 家公司中唯一反向），
          虚拟公司与其他真实公司全部同向，无误报。
    """
    ELEC_MIN = 10.0  # 电量单年变动门槛 %
    EMIS_MIN = 10.0  # 排放单年变动门槛 %
    fields = ["scope1_emission_tco2e", "scope2_market_based_tco2e", "scope1_2_emission_tco2e"]
    out: List[dict] = []
    for firm_id, g in metrics_df.groupby("firm_id"):
        g = g.sort_values("year")
        for i in range(1, len(g)):
            r0, r1 = g.iloc[i - 1], g.iloc[i]
            e0, e1 = r0.get("electricity_mwh"), r1.get("electricity_mwh")
            if pd.isna(e0) or pd.isna(e1) or e0 == 0:
                continue
            elec_pct = (e1 - e0) / e0 * 100.0
            if abs(elec_pct) < ELEC_MIN:
                continue
            for f in fields:
                v0, v1 = r0.get(f), r1.get(f)
                if pd.isna(v0) or pd.isna(v1) or v0 == 0:
                    continue
                emis_pct = (v1 - v0) / v0 * 100.0
                if abs(emis_pct) < EMIS_MIN:
                    continue
                if (emis_pct > 0) == (elec_pct > 0):
                    continue  # 同向变动，物理正常
                out.append({
                    "detection_channel": "structured",
                    "cohort": r1.get("cohort"),
                    "firm_id": firm_id,
                    "company_name": r1.get("company_name"),
                    "facility_id": None,
                    "facility_name": None,
                    "aoi_id": None,
                    "year": int(r1["year"]),
                    "image_date": None,
                    "variable": f,
                    "reported_value": v1,
                    "recalculated_value": v0,
                    "unit": "tCO2e",
                    "spatial_boundary": "",
                    "data_source": "unified_annual_metrics.csv (排放-电量脱钩)",
                    "calculation": (f"{int(r0['year'])}→{int(r1['year'])}: 电量 {elec_pct:+.1f}% "
                                    f"而 {f} {emis_pct:+.1f}%，方向相反，"
                                    f"排放轨迹与活动水平脱钩（疑似平滑）"),
                    "confidence": clip_conf(CONFIDENCE_BASE.get(
                        "emission_electricity_decoupling", 0.70)),
                    "conclusion": "emission_electricity_decoupling",
                    "notes": "直接排放随产量同向变动是物理常态；反向大幅变动提示排放序列被改写。",
                    "evidence_location": f"Year {int(r1['year'])}",
                    "report_file": "",
                })
    return out


# ============================================================
# 新增规则：基于 structured_claims.json 的三条验证规则
# ============================================================

def rule_structured_claim_vs_table(claim: dict, metrics_df: pd.DataFrame) -> Optional[dict]:
    """
    规则 A：结构化声称 vs 年度指标表
    输入：单条 structured 声称
    输出：若偏差 > 1% 则返回 finding，通道为 'text'
    """
    firm_id = claim["firm_id"]
    year = claim["year"]
    claim_type = claim["claim_type"]
    claim_val = claim["claim_value"]
    claim_unit = claim.get("claim_unit", "")

    if claim_val is None:
        return None

    # 从 unified_annual_metrics 中取对应值
    row = metrics_df[(metrics_df["firm_id"] == firm_id) & (metrics_df["year"] == year)]
    if row.empty:
        return None
    row = row.iloc[0]

    # 检查表中是否有该字段
    if claim_type not in row or pd.isna(row[claim_type]):
        return None
    table_val = row[claim_type]

    if table_val == 0:
        return None

    # 单位一致性检查：tce vs MWh 不强行换算
    if claim_unit == "tce" and "mwh" in str(claim_type).lower():
        # 不强行换算，跳过并记录单位不一致
        return None

    dev_pct = abs(claim_val - table_val) / table_val * 100.0
    if dev_pct > 1.0:  # 1% 容差
        return {
            "detection_channel": "text",
            "cohort": claim["cohort"],
            "firm_id": firm_id,
            "company_name": claim["company_name"],
            "facility_id": None,
            "facility_name": None,
            "aoi_id": None,
            "year": year,
            "image_date": None,
            "variable": claim_type,
            "reported_value": claim_val,
            "recalculated_value": table_val,
            "unit": claim_unit,
            "spatial_boundary": None,
            "data_source": f"structured_claims.json vs unified_annual_metrics (from {claim.get('report_file', 'unknown')})",
            "calculation": (f"报告叙述值 {claim_val:.2f} {claim_unit} vs 结构化表 {table_val:.2f}, "
                            f"偏差 {dev_pct:.2f}%"),
            "confidence": clip_conf(CONFIDENCE_BASE.get("narrative_table_mismatch", 0.75)),
            "conclusion": "narrative_table_mismatch",
            "notes": f"叙述段落与结构化表中的 {claim_type} 不一致，疑似手工编辑失误或刻意篡改。",
            "evidence_location": claim.get("evidence_location", ""),
            "report_file": claim.get("report_file", ""),
        }
    return None


def rule_satellite_claim_vs_proxy(claim: dict, facility_panel_df: pd.DataFrame) -> Optional[dict]:
    """
    规则 B：卫星声称 vs 遥感特征表
    输入：单条 satellite 声称（claims 无 facility_id，按 (firm_id, year) 解析设施）
    输出：若偏差 > 20% 则返回 finding，通道为 'satellite'
    """
    if claim["claim_value"] is None:
        return None

    firm_id = claim["firm_id"]
    year = claim["year"]
    claim_type = claim["claim_type"]
    claim_val = claim["claim_value"]

    # 解析设施：claims 无 facility_id 字段，单设施公司直接命中，
    # 多设施公司用 claim_text 中的设施名片段匹配
    fac_row = resolve_claim_facility(claim, facility_panel_df, year_col="year")
    if fac_row is None:
        return None
    facility_id = fac_row["facility_id"]

    # 从 facility_panel 中取同 (firm_id, facility_id, year) 的值
    row = facility_panel_df[
        (facility_panel_df["firm_id"] == firm_id) &
        (facility_panel_df["facility_id"] == facility_id) &
        (facility_panel_df["year"] == year)
    ]
    if row.empty:
        return None
    row = row.iloc[0]

    # 映射 claim_type → facility_panel 列名
    col_map = {
        # 评估建议 2：真实公司披露装机 vs GEE 屋顶容量代理（Sungrow 42 MW 案例）
        "rooftop_pv_installed_capacity_mw": "estimated_rooftop_pv_capacity_mw",
        "roof_pv_area_m2": "roof_pv_area_m2",
        "built_up_area_ha": "built_up_area_ha",
        "estimated_rooftop_pv_capacity_mw": "estimated_rooftop_pv_capacity_mw",
        "night_light": "annual_night_light_mean",
        "heat_index": "land_surface_heat_index",
        "ndvi": "ndvi_buffer_mean",
        "lst_c": "land_surface_temperature_c",
    }
    if claim_type not in col_map:
        return None
    proxy_col = col_map[claim_type]
    if proxy_col not in row or pd.isna(row[proxy_col]):
        return None
    proxy_val = row[proxy_col]

    if proxy_val == 0:
        return None

    dev_pct = abs(claim_val - proxy_val) / proxy_val * 100.0
    if dev_pct > 20.0:  # 20% 容差
        return {
            "detection_channel": "satellite",
            "cohort": claim["cohort"],
            "firm_id": firm_id,
            "company_name": claim["company_name"],
            "facility_id": row.get("facility_id"),
            "facility_name": row.get("facility_name"),
            "aoi_id": row.get("aoi_id"),
            "year": year,
            "image_date": row.get("image_date"),
            "variable": claim_type,
            "reported_value": claim_val,
            "recalculated_value": proxy_val,
            "unit": claim.get("claim_unit", ""),
            "spatial_boundary": row.get("aoi_bbox") or "AOI bbox",
            "data_source": f"structured_claims.json vs facility_panel (from {claim.get('report_file', 'unknown')})",
            "calculation": (f"声明 {claim_val:.2f} {claim.get('claim_unit', '')} vs 遥感 {proxy_val:.2f}, "
                            f"偏差 {dev_pct:.1f}%"),
            "confidence": clip_conf(CONFIDENCE_BASE.get("claim_pv_inflation", 0.70)),
            "conclusion": "satellite_claim_mismatch",
            "notes": f"文本声称的 {claim_type} 与遥感特征值不一致。",
            "evidence_location": claim.get("evidence_location", ""),
            "report_file": claim.get("report_file", ""),
        }
    return None


def rule_qualitative_claim_vs_trend(claim: dict, facility_trends_df: pd.DataFrame) -> Optional[dict]:
    """
    规则 C：定性声称 vs 遥感趋势
    输入：定性声称（含 claim_direction 和 facility_id）
    输出：若物理趋势与声称方向矛盾，则返回 finding，通道为 'satellite'
    """
    if claim.get("claim_direction") in [None, "unknown"]:
        return None
    if claim["claim_value"] is not None:
        # 数值型声称不应走定性规则
        return None

    firm_id = claim["firm_id"]

    # 解析设施（claims 无 facility_id 字段；趋势表每设施一行，按 firm_id 匹配）
    fac_row = resolve_claim_facility(claim, facility_trends_df)
    if fac_row is None:
        return None
    facility_id = fac_row["facility_id"]

    # 从 facility_trends 中取该设施的趋势
    row = facility_trends_df[
        (facility_trends_df["firm_id"] == firm_id) &
        (facility_trends_df["facility_id"] == facility_id)
    ]
    if row.empty:
        # 尝试用 facility_name 匹配
        fname = claim.get("facility_name", "")
        if fname:
            row = facility_trends_df[
                (facility_trends_df["firm_id"] == firm_id) &
                (facility_trends_df["facility_name"].str.contains(fname[:10], case=False, na=False))
            ]
            if row.empty:
                return None
        else:
            return None
    row = row.iloc[0]

    # 根据 claim_type 决定看哪个指标
    metric_map = {
        "heat_control": ("heat_delta", 0.5),
        "night_shift": ("nightlight_delta", 0.0),
        "emission_reduction": ("heat_delta", 0.5),
        "boundary_claim": (None, None),  # 边界声称不在此处处理
        "directional_claim": ("heat_delta", 0.5),
    }
    if claim["claim_type"] not in metric_map:
        return None
    metric_info = metric_map[claim["claim_type"]]
    if metric_info[0] is None:
        return None
    metric_col, threshold = metric_info

    if metric_col not in row or pd.isna(row[metric_col]):
        return None
    delta = row[metric_col]

    # 矛盾判断：热指数上升 > 阈值 但声称“控制”或“低”
    claim_dir = claim.get("claim_direction", "")
    if claim["claim_type"] == "heat_control" and claim_dir in ["control", "low", "decrease"]:
        if delta > threshold:
            return {
                "detection_channel": "satellite",
                "cohort": claim["cohort"],
                "firm_id": firm_id,
                "company_name": claim["company_name"],
                "facility_id": facility_id,
                "facility_name": row.get("facility_name"),
                "aoi_id": row.get("aoi_id_end"),
                "year": int(row.get("end_year", 2025)),
                "image_date": row.get("image_date_end"),
                "variable": "land_surface_heat_index_trend",
                "reported_value": None,
                "recalculated_value": delta,
                "unit": "°C",
                "spatial_boundary": row.get("aoi_bbox") or "AOI bbox",
                "data_source": f"structured_claims.json vs facility_trends (from {claim.get('report_file', 'unknown')})",
                "calculation": (f"热岛指数上升 {delta:.2f}°C，但文本声称 '{claim_dir}'"),
                "confidence": clip_conf(CONFIDENCE_BASE.get("heat_scope1_mismatch", 0.50)),
                "conclusion": "qualitative_trend_mismatch",
                "notes": f"物理趋势上升与定性声称方向矛盾。",
                "evidence_location": claim.get("evidence_location", ""),
                "report_file": claim.get("report_file", ""),
            }

    # 夜光判断：夜光上升但声称“低负荷/错峰”
    if claim["claim_type"] == "night_shift" and claim_dir in ["decrease", "low", "control"]:
        if delta > 0:
            return {
                "detection_channel": "satellite",
                "cohort": claim["cohort"],
                "firm_id": firm_id,
                "company_name": claim["company_name"],
                "facility_id": facility_id,
                "facility_name": row.get("facility_name"),
                "aoi_id": row.get("aoi_id_end"),
                "year": int(row.get("end_year", 2025)),
                "image_date": row.get("image_date_end"),
                "variable": "nightlight_trend",
                "reported_value": None,
                "recalculated_value": delta,
                "unit": "nW/cm2/sr",
                "spatial_boundary": row.get("aoi_bbox") or "AOI bbox",
                "data_source": f"structured_claims.json vs facility_trends (from {claim.get('report_file', 'unknown')})",
                "calculation": (f"夜光强度上升 {delta:.2f}，但文本声称 '{claim_dir}'"),
                "confidence": clip_conf(CONFIDENCE_BASE.get("nightlight_production_mismatch", 0.55)),
                "conclusion": "qualitative_trend_mismatch",
                "notes": f"夜光强度上升与低负荷/错峰声称矛盾。",
                "evidence_location": claim.get("evidence_location", ""),
                "report_file": claim.get("report_file", ""),
            }

    return None


# ============================================================
# 主流程
# ============================================================
def main():
    fac_trends_df = pd.read_csv(FACILITY_TRENDS_OUTPUT)
    comp_df = pd.read_csv(COMPANY_TRENDS_OUTPUT)
    metrics_df = pd.read_csv(UNIFIED_METRICS_OUTPUT)
    panel_df = pd.read_csv(FACILITY_PANEL)

    comp_lookup = comp_df.set_index(["cohort", "firm_id"], drop=False)

    findings: List[dict] = []
    reports_cache = {
        COHORT_VIRTUAL: load_reports(COHORT_VIRTUAL),
        COHORT_REAL: load_reports(COHORT_REAL),
    }

    # ---- 加载 structured_claims.json ----
    claims = []
    if os.path.exists(STRUCTURED_CLAIMS_PATH):
        try:
            with open(STRUCTURED_CLAIMS_PATH, "r", encoding="utf-8") as f:
                claims = json.load(f)
            print(f"[step5] 加载 structured_claims.json: {len(claims)} 条声称")
        except Exception as e:
            print(f"[step5] 警告: 无法加载 structured_claims.json: {e}")
            claims = []
    else:
        print(f"[step5] 警告: 未找到 structured_claims.json: {STRUCTURED_CLAIMS_PATH}")

    # ---- 原有设施级规则 ----
    for _, fac in fac_trends_df.iterrows():
        key = (fac["cohort"], fac["firm_id"])
        if key not in comp_lookup.index:
            continue
        comp = comp_lookup.loc[key]
        if isinstance(comp, pd.DataFrame):
            comp = comp.iloc[0]

        findings += rule_expansion_vs_energy(fac, comp)
        findings += rule_pv_growth_vs_renewable(fac, comp)
        findings += rule_nightlight_vs_production(fac, comp)
        findings += rule_pv_capacity_area_consistency(fac, comp)
        findings += rule_heat_vs_scope1(fac, comp)
        findings += rule_report_pv_claim_vs_remote(fac, reports_cache[fac["cohort"]])

    # ---- 原有公司级规则 ----
    for _, comp in comp_df.iterrows():
        findings += rule_energy_intensity_drop(comp)

    # ---- YoY 异常检测 ----
    if os.path.exists(YOY_DETAILS_FILE):
        yoy_df = pd.read_csv(YOY_DETAILS_FILE)
        findings += rule_yoy_anomaly_from_details(yoy_df)
    else:
        print(f"[step5] 未找到 {YOY_DETAILS_FILE}, 请先运行 step4")

    # ---- 排放-电量脱钩检测（识别排放轨迹平滑：LONGi S1 案例） ----
    findings += rule_emission_electricity_decoupling(metrics_df)

    # ---- 文本过程措辞扫描 ----
    for cohort in [COHORT_VIRTUAL, COHORT_REAL]:
        for _, comp in comp_df[comp_df["cohort"] == cohort].iterrows():
            firm_id = comp["firm_id"]
            findings += rule_process_wording(reports_cache[cohort], firm_id, cohort)

    # ---- 新增：基于 structured_claims.json 的三条规则 ----
    if claims:
        for claim in claims:
            # 跳过数据可用性声明（无数值）
            if claim.get("claim_type") == "data_availability_statement":
                continue

            # 规则 A：structured 声称 vs 指标表
            if claim.get("claim_category") == "structured" and claim.get("claim_value") is not None:
                result = rule_structured_claim_vs_table(claim, metrics_df)
                if result:
                    findings.append(result)

            # 规则 B：satellite 声称 vs 遥感特征
            # （另含真实公司披露装机 rooftop_pv_installed_capacity_mw —— 评估建议 2：
            #  Sungrow 披露 42 MW 与 GEE 屋顶容量代理的物理一致性比较）
            if (claim.get("claim_category") == "satellite"
                    or claim.get("claim_type") == "rooftop_pv_installed_capacity_mw") \
                    and claim.get("claim_value") is not None:
                result = rule_satellite_claim_vs_proxy(claim, panel_df)
                if result:
                    findings.append(result)

            # 规则 C：定性声称 vs 遥感趋势
            if claim.get("claim_category") == "qualitative":
                result = rule_qualitative_claim_vs_trend(claim, fac_trends_df)
                if result:
                    findings.append(result)

    # ---- 保存 ----
    df = pd.DataFrame(findings)
    if df.empty:
        print("[step5] 未触发任何规则")
    else:
        df.insert(0, "finding_id", [f"F{idx:04d}" for idx in range(1, len(df) + 1)])
    df.to_csv(FINDINGS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[step5] 原始 findings 已保存: {FINDINGS_OUTPUT}  行数={len(df)}")
    if not df.empty:
        print("\n=== 按检测通道统计 ===")
        print(df["detection_channel"].value_counts())
        print("\n=== 按结论类型统计 ===")
        print(df["conclusion"].value_counts())
    return df


if __name__ == "__main__":
    main()