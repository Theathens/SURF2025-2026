#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_pull_real_satellite.py
============================
仅针对 real cohort：AOI bbox 已在 facility_satellite_context_real.csv 中给出，
遥感字段全部为空。本步骤调用 GEE 直接在每个 (facility, year) 的 AOI 上
统计 6 个官方要求的变量：
  - built_up_area_ha         : Dynamic World 'built' 类概率 > 0.5 的像元面积
  - roof_pv_area_m2          : 高反照率 + 低 NDVI + built 掩膜的启发式估计
  - estimated_rooftop_pv_capacity_mw : roof_pv_area / 6000 m²/MW
  - annual_night_light_mean  : VIIRS 年均夜光辐射
  - ndvi_buffer_mean         : Sentinel-2 年内 NDVI 均值
  - land_surface_temperature_c / land_surface_heat_index : MODIS LST（换算℃）
                               heat_index = LST_facility - LST_周边缓冲

依赖:
    pip install earthengine-api geemap pandas
    先执行 `earthengine authenticate` 一次即可。

参考 confrontation2/step2 的 GEE 初始化、投影推算与限流方式。
"""
import os
import time
import math
import pandas as pd
import numpy as np
from typing import Optional

try:
    import ee
except ImportError:
    raise ImportError("需要 earthengine-api：pip install earthengine-api geemap")

from config import (
    REAL_CONTEXT_FILE, REAL_GEE_OUTPUT, GEE_CONFIG,
)

# ---------------- GEE 初始化 ----------------
def init_gee(project_id: str):
    try:
        ee.Initialize(project=project_id)
        print(f"[step2] GEE 已初始化 project={project_id}")
    except Exception:
        print("[step2] 尚未授权，正在打开浏览器登录……")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("[step2] 授权完成并已初始化")


# ---------------- 工具函数 ----------------
def parse_bbox(bbox_str: str) -> Optional[ee.Geometry]:
    """
    aoi_bbox 字段形如 'south=36.1246010;north=36.1329060;west=120.4093963;east=120.4199794'
    """
    if not isinstance(bbox_str, str) or "=" not in bbox_str:
        return None
    kv = {}
    for token in bbox_str.split(";"):
        token = token.strip()
        if "=" in token:
            k, v = token.split("=", 1)
            try:
                kv[k.strip()] = float(v.strip())
            except ValueError:
                pass
    for key in ("south", "north", "west", "east"):
        if key not in kv:
            return None
    return ee.Geometry.Rectangle([kv["west"], kv["south"], kv["east"], kv["north"]])


def year_window(year: int):
    return f"{year}-01-01", f"{year}-12-31"


def safe_get(dictionary: ee.Dictionary, key: str) -> Optional[float]:
    try:
        v = dictionary.get(key).getInfo()
        return None if v is None else float(v)
    except Exception:
        return None


# ---------------- 变量计算 ----------------
def compute_builtup_ha(aoi: ee.Geometry, year: int) -> Optional[float]:
    """Dynamic World 年内 built 概率均值 > 0.5 的像元面积（ha）。"""
    start, end = year_window(year)
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(aoi).filterDate(start, end))
    if dw.size().getInfo() == 0:
        return None
    built_prob = dw.select("built").mean()
    built_mask = built_prob.gt(0.5)
    pixel_area = ee.Image.pixelArea().updateMask(built_mask)
    area_m2 = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e10
    )
    v = safe_get(area_m2, "area")
    return None if v is None else v / 10000.0


def compute_roof_pv_m2(aoi: ee.Geometry, year: int) -> Optional[float]:
    """
    屋顶 PV 面积启发式：Sentinel-2 中值合成上，
      - Red-blue ratio 低（深色蓝色板）
      - NDVI < 0.15（非植被）
      - Dynamic World built 概率 > 0.5（在建筑物上）
    这只是遥感层近似，用作"再计算值"用于挑战企业披露。
    """
    start, end = year_window(year)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", GEE_CONFIG["cloud_thresh"])))
    if s2.size().getInfo() == 0:
        return None
    med = s2.median().divide(10000)
    ndvi = med.normalizedDifference(["B8", "B4"]).rename("ndvi")
    blue_over_red = med.select("B2").divide(med.select("B4").add(1e-6))
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(aoi).filterDate(start, end).select("built").mean())
    pv_mask = (blue_over_red.gt(1.05)
               .And(ndvi.lt(0.15))
               .And(dw.gt(0.4))
               .And(med.select("B2").gt(0.05))
               .And(med.select("B2").lt(0.35)))
    area = ee.Image.pixelArea().updateMask(pv_mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=10, maxPixels=1e10
    )
    return safe_get(area, "area")


def compute_night_light(aoi: ee.Geometry, year: int) -> Optional[float]:
    start, end = year_window(year)
    coll = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterBounds(aoi).filterDate(start, end).select("avg_rad"))
    if coll.size().getInfo() == 0:
        return None
    mean_img = coll.mean()
    stat = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=500, maxPixels=1e10
    )
    return safe_get(stat, "avg_rad")


def compute_ndvi_mean(aoi: ee.Geometry, year: int) -> Optional[float]:
    start, end = year_window(year)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", GEE_CONFIG["cloud_thresh"])))
    if s2.size().getInfo() == 0:
        return None
    med = s2.median()
    ndvi = med.normalizedDifference(["B8", "B4"]).rename("ndvi")
    stat = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=10, maxPixels=1e10
    )
    return safe_get(stat, "ndvi")


def compute_lst_and_heat(aoi: ee.Geometry, year: int):
    """
    MODIS LST 年均，转 ℃；
    heat_index = LST(AOI) - LST(周边 3km buffer 减 AOI)
    """
    start, end = year_window(year)
    coll = (ee.ImageCollection("MODIS/061/MOD11A2")
            .filterDate(start, end).select("LST_Day_1km"))
    if coll.size().getInfo() == 0:
        return None, None
    lst_k = coll.mean().multiply(0.02)  # scale factor
    lst_c = lst_k.subtract(273.15).rename("lst_c")
    aoi_stat = lst_c.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=1000, maxPixels=1e10
    )
    aoi_lst = safe_get(aoi_stat, "lst_c")
    ring = aoi.buffer(3000).difference(aoi, maxError=10)
    ring_stat = lst_c.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=ring, scale=1000, maxPixels=1e10
    )
    ring_lst = safe_get(ring_stat, "lst_c")
    if aoi_lst is None or ring_lst is None:
        return aoi_lst, None
    return aoi_lst, aoi_lst - ring_lst


# ---------------- 主流程 ----------------
def process_row(row: pd.Series) -> dict:
    aoi = parse_bbox(row.get("aoi_bbox", ""))
    if aoi is None:
        # 若 bbox 缺失，退化为以点为中心的 500m buffer
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            aoi = ee.Geometry.Point([lon, lat]).buffer(500)
        except Exception:
            return {}
    year = int(row["year"])

    built = compute_builtup_ha(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])

    pv_m2 = compute_roof_pv_m2(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])
    pv_mw = pv_m2 / 6000.0 if pv_m2 is not None else None

    nl = compute_night_light(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])

    ndvi = compute_ndvi_mean(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])

    lst_c, heat = compute_lst_and_heat(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])

    return {
        "cohort": row.get("cohort", "real_public_reference"),
        "firm_id": row["firm_id"],
        "facility_id": row["facility_id"],
        "aoi_id": row["aoi_id"],
        "year": year,
        "built_up_area_ha_recalc": built,
        "roof_pv_area_m2_recalc": pv_m2,
        "estimated_rooftop_pv_capacity_mw_recalc": pv_mw,
        "annual_night_light_mean_recalc": nl,
        "ndvi_buffer_mean_recalc": ndvi,
        "land_surface_temperature_c_recalc": lst_c,
        "land_surface_heat_index_recalc": heat,
        "gee_image_date": f"{year}-06-30",
        "gee_source": "S2_SR_HARMONIZED; DynamicWorld V1; VIIRS DNB; MODIS/061/MOD11A2",
    }


def main():
    init_gee(GEE_CONFIG["project_id"])
    df = pd.read_csv(REAL_CONTEXT_FILE)
    print(f"[step2] 读取 real cohort AOI 记录 {len(df)} 条")

    out = []
    for idx, row in df.iterrows():
        try:
            r = process_row(row)
            if r:
                out.append(r)
                print(f"  ✅ {row['facility_id']} {row['year']}: built={r['built_up_area_ha_recalc']}, "
                      f"pv={r['roof_pv_area_m2_recalc']}, nl={r['annual_night_light_mean_recalc']}")
        except Exception as e:
            print(f"  ❌ {row['facility_id']} {row['year']}: {e}")

    out_df = pd.DataFrame(out)
    os.makedirs(os.path.dirname(REAL_GEE_OUTPUT), exist_ok=True)
    out_df.to_csv(REAL_GEE_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[step2] real cohort GEE 再计算已保存: {REAL_GEE_OUTPUT}")
    return out_df


if __name__ == "__main__":
    main()
