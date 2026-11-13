# step1_load_structured.py
"""
Step 1: 分别加载 virtual / real cohort 的 facility 遥感表和年度 ESG 指标表，
        对齐字段、打上 cohort 标签、落盘为统一格式，供后续 step 消费。

- virtual cohort 已经包含遥感字段（built_up_area_ha 等）
- real cohort 只有 AOI 坐标 + bbox，遥感字段全空，需要 step2 从 GEE 补齐
"""
import os
import pandas as pd
import numpy as np

from config import (
    VIRTUAL_PROXY_FILE, VIRTUAL_METRICS_FILE,
    REAL_CONTEXT_FILE, REAL_METRICS_FILE,
    UNIFIED_PROXY_OUTPUT, UNIFIED_METRICS_OUTPUT,
    COHORT_VIRTUAL, COHORT_REAL,
)

# 统一后遥感表使用的列
PROXY_COLS = [
    "cohort", "firm_id", "company_name", "facility_id", "facility_name", "year",
    "aoi_id", "country", "province", "city", "latitude", "longitude", "image_date",
    "aoi_area_ha", "aoi_bbox",
    "built_up_area_ha", "roof_pv_area_m2", "estimated_rooftop_pv_capacity_mw",
    "annual_night_light_mean", "ndvi_buffer_mean",
    "land_surface_temperature_c", "land_surface_heat_index",
    "satellite_data_source", "proxy_status", "quality_flag",
]

# 统一后年报指标表使用的列（保留全部字段并新增 cohort）
NUMERIC_METRIC_COLS = [
    "revenue_million_cny", "production_volume", "employee_count", "factories_count",
    "total_energy_mwh", "electricity_mwh", "purchased_electricity_mwh",
    "self_generated_pv_mwh", "green_power_purchased_mwh", "green_certificates_mwh",
    "renewable_electricity_mwh", "renewable_electricity_ratio_pct",
    "natural_gas_mwh", "diesel_mwh",
    "scope1_emission_tco2e", "scope2_market_based_tco2e",
    "scope1_2_emission_tco2e", "scope3_emission_tco2e",
    "emissions_intensity_tco2e_per_production_unit",
    "water_withdrawal_m3", "solid_waste_tonnes",
    "direct_energy_mwh", "energy_intensity_mwh_per_production_unit",
]


def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_virtual_proxy() -> pd.DataFrame:
    df = pd.read_csv(VIRTUAL_PROXY_FILE)
    df["cohort"] = COHORT_VIRTUAL
    # virtual 表没有 aoi_bbox 列，占位
    if "aoi_bbox" not in df.columns:
        df["aoi_bbox"] = np.nan
    numeric = [
        "aoi_area_ha", "built_up_area_ha", "roof_pv_area_m2",
        "estimated_rooftop_pv_capacity_mw", "annual_night_light_mean",
        "ndvi_buffer_mean", "land_surface_temperature_c",
        "land_surface_heat_index", "latitude", "longitude",
    ]
    df = _to_num(df, numeric)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in PROXY_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[PROXY_COLS]


def load_real_context() -> pd.DataFrame:
    df = pd.read_csv(REAL_CONTEXT_FILE)
    df["cohort"] = COHORT_REAL
    numeric = [
        "aoi_area_ha", "built_up_area_ha", "roof_pv_area_m2",
        "estimated_rooftop_pv_capacity_mw", "annual_night_light_mean",
        "ndvi_buffer_mean", "land_surface_temperature_c",
        "land_surface_heat_index", "latitude", "longitude",
    ]
    df = _to_num(df, numeric)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in PROXY_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[PROXY_COLS]


def load_virtual_metrics() -> pd.DataFrame:
    df = pd.read_csv(VIRTUAL_METRICS_FILE)
    if "cohort" not in df.columns:
        df["cohort"] = COHORT_VIRTUAL
    df = _to_num(df, NUMERIC_METRIC_COLS)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def load_real_metrics() -> pd.DataFrame:
    df = pd.read_csv(REAL_METRICS_FILE)
    if "cohort" not in df.columns:
        df["cohort"] = COHORT_REAL
    df = _to_num(df, NUMERIC_METRIC_COLS)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def unify_and_dump():
    virtual_proxy = load_virtual_proxy()
    real_proxy = load_real_context()
    proxy = pd.concat([virtual_proxy, real_proxy], ignore_index=True)
    proxy = proxy.sort_values(["cohort", "firm_id", "facility_id", "year"])
    proxy.to_csv(UNIFIED_PROXY_OUTPUT, index=False, encoding="utf-8-sig")

    virtual_metrics = load_virtual_metrics()
    real_metrics = load_real_metrics()
    metrics = pd.concat([virtual_metrics, real_metrics], ignore_index=True)
    metrics = metrics.sort_values(["cohort", "firm_id", "year"])
    metrics.to_csv(UNIFIED_METRICS_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"[step1] 统一 facility proxy 已写出: {UNIFIED_PROXY_OUTPUT}  行数={len(proxy)}")
    print(f"[step1] 统一年报指标已写出: {UNIFIED_METRICS_OUTPUT}  行数={len(metrics)}")
    print("\n=== cohort × facility 计数 ===")
    print(proxy.groupby("cohort")["facility_id"].nunique())
    return proxy, metrics


if __name__ == "__main__":
    unify_and_dump()
