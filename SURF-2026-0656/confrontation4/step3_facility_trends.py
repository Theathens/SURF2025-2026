# step3_facility_trends.py
"""
Step 3: 计算每个设施的年际趋势。
输入:
  - unified_facility_proxy.csv (来自 step1)
  - real_gee_proxy.csv         (来自 step2, 可选)
逻辑:
  - real cohort 若已有 step2 的 GEE 重算值, 用重算值覆盖原本为空的字段
  - 对每个 facility 分别记录 逐年面板 + 首末年趋势
输出:
  - facility_trends.csv        首末年趋势 (供 step5)
  - facility_panel.csv         年度面板 (供 step6 findings 引用)
"""
import os
import pandas as pd
import numpy as np

from config import (
    UNIFIED_PROXY_OUTPUT, REAL_GEE_OUTPUT,
    FACILITY_TRENDS_OUTPUT, DATA_DIR, COHORT_REAL,
)

RECALC_MAP = {
    "built_up_area_ha_recalc": "built_up_area_ha",
    "roof_pv_area_m2_recalc": "roof_pv_area_m2",
    "estimated_rooftop_pv_capacity_mw_recalc": "estimated_rooftop_pv_capacity_mw",
    "annual_night_light_mean_recalc": "annual_night_light_mean",
    "ndvi_buffer_mean_recalc": "ndvi_buffer_mean",
    "land_surface_temperature_c_recalc": "land_surface_temperature_c",
    "land_surface_heat_index_recalc": "land_surface_heat_index",
}

FACILITY_PANEL_OUTPUT = os.path.join(DATA_DIR, "facility_panel.csv")


def merge_real_gee(proxy: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(REAL_GEE_OUTPUT):
        print("[step3] 未发现 real_gee_proxy.csv, real cohort 的遥感字段将保持原样")
        return proxy
    gee = pd.read_csv(REAL_GEE_OUTPUT)
    key = ["firm_id", "facility_id", "year"]
    merged = proxy.merge(gee, on=key, how="left", suffixes=("", "_gee"))
    for src, dst in RECALC_MAP.items():
        if src in merged.columns:
            merged[dst] = merged[dst].where(merged[dst].notna(), merged[src])
    if "gee_image_date" in merged.columns:
        merged["image_date"] = merged["image_date"].where(
            merged["image_date"].notna(), merged["gee_image_date"]
        )
    if "gee_source" in merged.columns:
        merged.loc[merged["cohort"] == COHORT_REAL, "satellite_data_source"] = (
            merged.loc[merged["cohort"] == COHORT_REAL, "gee_source"]
            .where(merged["gee_source"].notna(),
                   merged.loc[merged["cohort"] == COHORT_REAL, "satellite_data_source"])
        )
    return merged


def _pct(delta, start):
    if pd.isna(delta) or pd.isna(start) or start == 0:
        return np.nan
    return delta / start * 100.0


def compute_trends(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, firm, fac), g in panel.groupby(["cohort", "firm_id", "facility_id"]):
        g = g.sort_values("year")
        g_valid = g.dropna(subset=["year"])
        if len(g_valid) < 2:
            continue
        first, last = g_valid.iloc[0], g_valid.iloc[-1]

        def d(col):
            a, b = first.get(col), last.get(col)
            if pd.isna(a) or pd.isna(b):
                return np.nan, np.nan, a, b
            return b - a, _pct(b - a, a), a, b

        built_delta, built_pct, built_start, built_end = d("built_up_area_ha")
        pv_delta, pv_pct, pv_start, pv_end = d("roof_pv_area_m2")
        cap_delta, cap_pct, cap_start, cap_end = d("estimated_rooftop_pv_capacity_mw")
        nl_delta, nl_pct, nl_start, nl_end = d("annual_night_light_mean")
        ndvi_delta, ndvi_pct, ndvi_start, ndvi_end = d("ndvi_buffer_mean")
        heat_delta, heat_pct, heat_start, heat_end = d("land_surface_heat_index")
        lst_delta, _, lst_start, lst_end = d("land_surface_temperature_c")

        rows.append({
            "cohort": cohort,
            "firm_id": firm,
            "facility_id": fac,
            "company_name": first.get("company_name"),
            "facility_name": first.get("facility_name"),
            "aoi_id_start": first.get("aoi_id"),
            "aoi_id_end": last.get("aoi_id"),
            "start_year": int(first["year"]),
            "end_year": int(last["year"]),
            "image_date_start": first.get("image_date"),
            "image_date_end": last.get("image_date"),
            "latitude": first.get("latitude"),
            "longitude": first.get("longitude"),
            "aoi_bbox": first.get("aoi_bbox"),
            "spatial_boundary_hint": first.get("aoi_boundary_type") if "aoi_boundary_type" in first else None,

            "built_start_ha": built_start,
            "built_end_ha": built_end,
            "built_delta_ha": built_delta,
            "built_pct_change": built_pct,

            "pv_start_m2": pv_start,
            "pv_end_m2": pv_end,
            "pv_delta_m2": pv_delta,
            "pv_pct_change": pv_pct,

            "pv_capacity_start_mw": cap_start,
            "pv_capacity_end_mw": cap_end,
            "pv_capacity_delta_mw": cap_delta,
            "pv_capacity_pct_change": cap_pct,

            "nightlight_start": nl_start,
            "nightlight_end": nl_end,
            "nightlight_delta": nl_delta,
            "nightlight_pct_change": nl_pct,

            "ndvi_start": ndvi_start,
            "ndvi_end": ndvi_end,
            "ndvi_delta": ndvi_delta,

            "lst_start_c": lst_start,
            "lst_end_c": lst_end,
            "heat_start": heat_start,
            "heat_end": heat_end,
            "heat_delta": heat_delta,
        })
    return pd.DataFrame(rows)


def main():
    proxy = pd.read_csv(UNIFIED_PROXY_OUTPUT)
    proxy = merge_real_gee(proxy)
    proxy.to_csv(FACILITY_PANEL_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[step3] 年度面板已保存: {FACILITY_PANEL_OUTPUT}")

    trends = compute_trends(proxy)
    trends.to_csv(FACILITY_TRENDS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[step3] 设施趋势已保存: {FACILITY_TRENDS_OUTPUT}  行数={len(trends)}")
    print(trends[["cohort", "facility_id", "built_pct_change",
                  "pv_pct_change", "nightlight_delta"]].to_string(index=False))
    return trends


if __name__ == "__main__":
    main()
