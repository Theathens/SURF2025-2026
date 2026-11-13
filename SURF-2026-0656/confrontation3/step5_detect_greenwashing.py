# step5_detect_greenwashing.py
"""
Step 5: 规则库交叉验证。
输入:
  - facility_trends.csv  (step3)
  - facility_panel.csv   (step3)
  - company_trends.csv   (step4)
  - unified_annual_metrics.csv (step1)
  - 各 cohort 的 ESG 报告 md (可选, 用于文本层匹配)
输出:
  - raw_findings.csv     统一的原始发现表, 每行含 detection_channel ∈
                         {satellite, structured, text}, 具备三张模板所需的所有列
"""
import os
import re
import pandas as pd
import numpy as np
from typing import List, Optional

from config import (
    FACILITY_TRENDS_OUTPUT, COMPANY_TRENDS_OUTPUT,
    UNIFIED_METRICS_OUTPUT, FINDINGS_OUTPUT, DATA_DIR,
    VIRTUAL_REPORTS_DIR, REAL_REPORTS_DIR,
    THRESHOLDS, CONFIDENCE_BASE, DEFAULT_CONFIDENCE,
    COHORT_VIRTUAL, COHORT_REAL,
)

FACILITY_PANEL = os.path.join(DATA_DIR, "facility_panel.csv")


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
def _base(record: dict, fac: pd.Series, comp: pd.Series) -> dict:
    record.update({
        "cohort": fac["cohort"],
        "firm_id": fac["firm_id"],
        "company_name": fac.get("company_name") or comp.get("company_name"),
        "facility_id": fac["facility_id"],
        "facility_name": fac.get("facility_name"),
    })
    return record


# ------------------ 规则 ------------------
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
    """PV 面积 → 名义装机的自洽性 (6000 m²/MW)."""
    out = []
    area = fac.get("pv_end_m2")
    cap = fac.get("pv_capacity_end_mw")
    if pd.isna(area) or pd.isna(cap) or cap == 0:
        return out
    implied = area / THRESHOLDS["pv_area_to_mw_ratio"]
    if cap == 0:
        return out
    dev = (cap - implied) / cap * 100.0
    if abs(dev) > THRESHOLDS["pv_capacity_dev_pct"]:
        conf = CONFIDENCE_BASE["pv_capacity_area_mismatch"] + min(0.15, abs(dev) / 200.0)
        rec = _base({
            "detection_channel": "satellite",
            "aoi_id": fac.get("aoi_id_end"),
            "year": int(fac["end_year"]),
            "image_date": fac.get("image_date_end"),
            "variable": "estimated_rooftop_pv_capacity_mw",
            "reported_value": cap,
            "recalculated_value": round(implied, 3),
            "unit": "MW",
            "spatial_boundary": fac.get("aoi_bbox") or "AOI bbox",
            "data_source": "roof_pv_area_m2 / 6000 m² per MW",
            "calculation": (f"由 PV 面积 {area:.0f} m² 推算装机 ≈ {implied:.2f} MW, "
                            f"披露 {cap:.2f} MW, 偏差 {dev:.1f}%."),
            "confidence": clip_conf(conf),
            "conclusion": "pv_capacity_area_mismatch",
            "notes": "面积→装机换算与披露的 rooftop_pv_capacity_mw 不自洽.",
        }, fac, comp)
        out.append(rec)
    return out


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
    # 找出提及了 facility 名称片段的段落
    hits = []
    for para in re.split(r"\n\s*\n", text):
        p_lc = para.lower()
        if name_lc and any(tok in p_lc for tok in [name_lc, name_lc.split()[0] if name_lc else ""]):
            hits.append(para)
    if not hits:
        # 兜底：取全部提及 PV/光伏 的段落
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


# ------------------ 主流程 ------------------
def main():
    fac_df = pd.read_csv(FACILITY_TRENDS_OUTPUT)
    comp_df = pd.read_csv(COMPANY_TRENDS_OUTPUT)
    comp_lookup = comp_df.set_index(["cohort", "firm_id"], drop=False)

    findings: List[dict] = []
    reports_cache = {
        COHORT_VIRTUAL: load_reports(COHORT_VIRTUAL),
        COHORT_REAL: load_reports(COHORT_REAL),
    }

    for _, fac in fac_df.iterrows():
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

    for _, comp in comp_df.iterrows():
        findings += rule_energy_intensity_drop(comp)

    df = pd.DataFrame(findings)
    if df.empty:
        print("[step5] 未触发任何规则")
    else:
        df.insert(0, "finding_id", [f"F{idx:04d}" for idx in range(1, len(df) + 1)])
    df.to_csv(FINDINGS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[step5] 原始 findings 已保存: {FINDINGS_OUTPUT}  行数={len(df)}")
    if not df.empty:
        print(df["conclusion"].value_counts())
    return df


if __name__ == "__main__":
    main()
