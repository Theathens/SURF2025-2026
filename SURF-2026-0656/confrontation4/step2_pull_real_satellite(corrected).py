#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_pull_real_satellite.py (带图像保存)
============================
仅针对 real cohort：AOI bbox 已在 facility_satellite_context_real.csv 中给出，
遥感字段全部为空。本步骤调用 GEE 直接在每个 (facility, year) 的 AOI 上
统计 6 个官方要求的变量，并保存 RGB 缩略图供人工核验。

新增功能：
  - 每处理一个 (facility, year)，自动导出 512x512 RGB 缩略图 PNG
  - 图片保存在 real_gee_cache/images/ 下
  - 如果图片已存在则跳过（断点续传）

依赖:
    pip install earthengine-api geemap pandas rasterio pillow
    先执行 `earthengine authenticate` 一次即可。
"""
import os
import time
import tempfile
import pandas as pd
import numpy as np
import rasterio
from PIL import Image as PILImage
from typing import Optional

try:
    import ee
    import geemap
except ImportError:
    raise ImportError("需要 earthengine-api 和 geemap：pip install earthengine-api geemap")

from config import (
    REAL_CONTEXT_FILE, REAL_GEE_OUTPUT, GEE_CONFIG,
)

# ==================== 图像导出配置 ====================
IMAGE_EXPORT_CONFIG = {
    "enabled": True,                     # 是否启用图像导出
    "scale": 10,                         # 分辨率 (m)
    "image_size": 512,                   # 导出图片尺寸 (像素)
    "cloud_thresh": 30,                  # 云量阈值
    "output_subdir": "images",           # 子目录名
}
# =====================================================


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


# ============================================================
# 新增：图像导出函数（参考旧版 step2）
# ============================================================
def export_rgb_thumb(
    aoi: ee.Geometry,
    facility_name: str,
    year: int,
    output_dir: str,
    scale: int = 10,
    image_size: int = 512,
    cloud_thresh: int = 30,
) -> Optional[str]:
    """
    导出 AOI 的 RGB 缩略图 PNG，用于人工核验。

    与旧版 step2 的区别：
      - 输出固定为 512×512 像素（便于快速浏览）
      - 使用夏季（6-9月）无云中值合成
      - 自动跳过已存在的文件
    """
    safe_name = facility_name.replace("/", "_").replace(" ", "_").replace("，", "").replace(",", "")
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{safe_name}_{year}.png")

    # 断点续传：如果图片已存在，直接跳过
    if os.path.exists(image_path):
        return image_path

    # 构建夏季合成影像
    start_date = f"{year}-06-01"
    end_date = f"{year}-09-30"

    try:
        # 优先 Sentinel-2
        s2_col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        )

        def mask_s2_clouds(img):
            qa = img.select("QA60")
            cloud_bit = 1 << 10
            cirrus_bit = 1 << 11
            mask = (qa.bitwiseAnd(cloud_bit).eq(0)).And(qa.bitwiseAnd(cirrus_bit).eq(0))
            return img.updateMask(mask)

        s2_count = s2_col.size().getInfo()

        if s2_count > 0:
            composite = s2_col.map(mask_s2_clouds).median().clip(aoi)
            rgb = composite.select(["B4", "B3", "B2"])
        else:
            # 回退到 Landsat-8/9
            ls_col = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
                .filterBounds(aoi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUD_COVER", cloud_thresh))
            )

            def mask_ls_clouds(img):
                qa = img.select("QA_PIXEL")
                cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)
                return img.updateMask(cloud_mask)

            ls_count = ls_col.size().getInfo()
            if ls_count == 0:
                print(f"   ⚠️  无可用影像，跳过图片导出: {year}")
                return None

            composite = ls_col.map(mask_ls_clouds).median().clip(aoi)
            rgb = composite.select(["SR_B4", "SR_B3", "SR_B2"])

        # 转为 0-255 uint8
        rgb_uint8 = rgb.unitScale(0, 3000).multiply(255).clamp(0, 255).uint8()

        # 导出为 GeoTIFF → 转为 PNG
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_tif = os.path.join(tmpdir, "img.tif")
            geemap.ee_export_image(
                rgb_uint8,
                filename=tmp_tif,
                scale=scale,
                region=aoi,
                file_per_band=False,
                crs="EPSG:3857",  # Web Mercator，便于查看
            )

            with rasterio.open(tmp_tif) as src:
                arr = src.read()
                if arr.shape[0] != 3:
                    return None
                # 调整到固定尺寸
                arr = np.transpose(arr, (1, 2, 0))
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                img = PILImage.fromarray(arr, mode="RGB")
                img = img.resize((image_size, image_size), PILImage.Resampling.LANCZOS)
                img.save(image_path)

        return image_path

    except Exception as e:
        # 图像导出失败不影响主流程
        print(f"   ⚠️  图片导出失败 ({year}): {e}")
        return None


# ---------------- 主流程（带断点续传） ----------------
OUTPUT_COLUMNS = [
    "cohort", "firm_id", "facility_id", "aoi_id", "year",
    "built_up_area_ha_recalc", "roof_pv_area_m2_recalc",
    "estimated_rooftop_pv_capacity_mw_recalc",
    "annual_night_light_mean_recalc", "ndvi_buffer_mean_recalc",
    "land_surface_temperature_c_recalc", "land_surface_heat_index_recalc",
    "gee_image_date", "gee_source",
]


def _row_key(facility_id, aoi_id, year) -> tuple:
    try:
        y = int(float(year))
    except Exception:
        y = str(year)
    return (str(facility_id), str(aoi_id), y)


def load_completed_keys(output_path: str) -> set:
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()
    try:
        df = pd.read_csv(output_path, encoding="utf-8-sig")
    except Exception:
        return set()
    if df.empty:
        return set()
    keys = set()
    for _, r in df.iterrows():
        keys.add(_row_key(r.get("facility_id"), r.get("aoi_id"), r.get("year")))
    return keys


def append_row_to_csv(row: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    pd.DataFrame([row], columns=OUTPUT_COLUMNS).to_csv(
        output_path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig",
    )


def process_row_with_image(row: pd.Series, image_output_dir: Optional[str] = None) -> dict:
    """处理一行，同时导出图片（如果启用）"""
    aoi = parse_bbox(row.get("aoi_bbox", ""))
    if aoi is None:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            aoi = ee.Geometry.Point([lon, lat]).buffer(500)
        except Exception:
            return {}
    year = int(row["year"])

    # ---- 计算统计值 ----
    built = compute_builtup_ha(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])

    pv_m2 = compute_roof_pv_m2(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])
    # 使用统一的 15000 m²/MW 换算系数（与 config.py 一致）
    pv_mw = pv_m2 / 15000.0 if pv_m2 is not None else None

    nl = compute_night_light(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])

    ndvi = compute_ndvi_mean(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])

    lst_c, heat = compute_lst_and_heat(aoi, year)
    time.sleep(GEE_CONFIG["request_interval"])

    # ---- 导出图片（新增） ----
    if IMAGE_EXPORT_CONFIG["enabled"] and image_output_dir:
        facility_name = row.get("facility_name", f"facility_{row.get('facility_id', '')}")
        export_rgb_thumb(
            aoi=aoi,
            facility_name=facility_name,
            year=year,
            output_dir=image_output_dir,
            scale=IMAGE_EXPORT_CONFIG["scale"],
            image_size=IMAGE_EXPORT_CONFIG["image_size"],
            cloud_thresh=IMAGE_EXPORT_CONFIG["cloud_thresh"],
        )

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
    # ========== 第一步：完全不依赖网络的本地检查 ==========
    # 读取输入 CSV
    if not os.path.exists(REAL_CONTEXT_FILE):
        print(f"[step2] 错误：找不到输入文件 {REAL_CONTEXT_FILE}")
        return
    df = pd.read_csv(REAL_CONTEXT_FILE, encoding="utf-8-sig")
    total = len(df)
    print(f"[step2] 读取 real cohort AOI 记录 {total} 条")

    # 加载已完成的键（来自输出 CSV）
    completed = load_completed_keys(REAL_GEE_OUTPUT)
    pending = total - len(completed)
    print(f"[step2] 本地已完成 {len(completed)} 条，待处理 {pending} 条")

    # ========== 第二步：如果全部完成，直接退出（无需任何网络） ==========
    if pending <= 0:
        print("[step2] ✅ 所有任务已在本地完成（CSV 已存在），无需连接 GEE，程序退出。")
        return

    # ========== 第三步：确实有未完成的任务，此时才连接 GEE（需要 VPN） ==========
    init_gee(GEE_CONFIG["project_id"])

    # ========== 第四步：准备图片输出目录 ==========
    image_output_dir = None
    if IMAGE_EXPORT_CONFIG["enabled"]:
        base_dir = GEE_CONFIG.get("output_dir", os.path.dirname(REAL_GEE_OUTPUT))
        image_output_dir = os.path.join(base_dir, IMAGE_EXPORT_CONFIG["output_subdir"])
        os.makedirs(image_output_dir, exist_ok=True)
        print(f"[step2] 图片将保存至: {image_output_dir}")

    # ========== 第五步：循环处理未完成的任务 ==========
    processed_this_run = 0
    skipped = 0
    failed = 0

    try:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            key = _row_key(row.get("facility_id"), row.get("aoi_id"), row.get("year"))
            if key in completed:
                skipped += 1
                continue

            try:
                r = process_row_with_image(row, image_output_dir)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failed += 1
                print(f"  ❌ [{i}/{total}] {row.get('facility_id')} {row.get('year')}: {e}")
                continue

            if not r:
                failed += 1
                print(f"  ❌ [{i}/{total}] {row.get('facility_id')} {row.get('year')}: 无有效 AOI")
                continue

            append_row_to_csv(r, REAL_GEE_OUTPUT)
            completed.add(key)
            processed_this_run += 1
            print(f"  ✅ [{i}/{total}] {row['facility_id']} {row['year']}: "
                  f"built={r['built_up_area_ha_recalc']}, "
                  f"pv={r['roof_pv_area_m2_recalc']}, "
                  f"nl={r['annual_night_light_mean_recalc']}")

    except KeyboardInterrupt:
        print(f"\n[step2] 已捕获中断信号。本次运行新增 {processed_this_run} 条,"
              f"累计完成 {len(completed)}/{total}。重新运行即可从断点继续。")
        raise

    print(f"[step2] 完成:新增 {processed_this_run} 条,跳过 {skipped} 条,失败 {failed} 条。")
    print(f"[step2] CSV 已保存: {REAL_GEE_OUTPUT}")
    if image_output_dir:
        print(f"[step2] 图片已保存至: {image_output_dir}")
    return pd.read_csv(REAL_GEE_OUTPUT, encoding="utf-8-sig")


if __name__ == "__main__":
    main()



