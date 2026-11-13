#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_download_sentinel2.py
============================
基于 GEE 工业数据管线验证脚本重构的批量下载模块

功能：
  1. 读取 step1 生成的 cleaned_facility_list.csv
  2. 对每个厂区坐标，自动下载 T0（基准年）和 T1（检测年）的 Sentinel-2 影像
  3. 使用原始管线中验证过的投影对齐方法，确保影像尺寸正确
  4. 影像尺寸固定为 1280m × 1280m（对应 128 像素，匹配 U-Net 输入）

依赖：
  pip install earthengine-api geemap pandas rasterio pillow
"""

import os
import sys
import tempfile
import time
import pandas as pd
import numpy as np
import rasterio
from PIL import Image as PILImage
from typing import Optional, Tuple

# ── GEE 依赖 ──────────────────────────────────────────────────
try:
    import ee
    import geemap
except ImportError:
    raise ImportError("请先安装依赖: pip install earthengine-api geemap pandas rasterio pillow")

# ============================================================
# 1. 硬编码配置区（所有需要手动输入的内容都在这里）
# ============================================================
CONFIG = {
    # GEE Cloud Project ID（必改！）
    "project_id": "silent-elevator-500103-j4",

    # 厂区列表 CSV（由 step1 生成）
    "facility_csv": r"D:\Pycharm Projects\torch\confrontation\data\cleaned_facility_list.csv",

    # 输出目录（影像将保存在此目录下的 image/ 子目录中）
    "output_dir": r"D:\Pycharm Projects\torch\confrontation\data\sentinel2_aoi",

    # AOI 边长（米），默认 1280m = 128像素 × 10m 分辨率
    "aoi_size_meters": 1280,

    # 云量阈值（%），高于此值的影像会被剔除
    "cloud_thresh": 20,

    # 请求间隔（秒），避免 GEE 限流
    "request_interval": 2,
}


# ============================================================


# ============================================================
# 2. GEE 初始化与投影处理（直接移植自原始管线）
# ============================================================
def init_gee(project_id: str):
    """初始化 Earth Engine（原始管线已验证）"""
    try:
        ee.Initialize(project=project_id)
        print(f"✅ GEE 已初始化, project={project_id}")
    except Exception:
        print("尚未授权，正在打开浏览器进行一次性登录授权...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print(f"✅ GEE 授权完成并已初始化, project={project_id}")


def get_utm_epsg(lat: float, lon: float) -> str:
    """根据经纬度推算 UTM 投影 EPSG 码（回退方案）"""
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + utm_zone
    else:
        epsg = 32700 + utm_zone
    return f"EPSG:{epsg}"


# ============================================================
# 修改后的 get_s2_projection 函数
# ============================================================
def get_s2_projection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_thresh: int = 20
) -> Tuple[Optional[ee.Projection], Optional[str]]:
    """
    获取 Sentinel-2 原生投影，若无可用影像或出错则返回 (None, None)
    """
    try:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        )
        size = collection.size().getInfo()
        if size == 0:
            print(f"   ⚠️  {start_date}~{end_date} 无可用影像，将使用 UTM 推算")
            return None, None

        s2 = collection.first()
        proj = s2.select("B2").projection()
        proj_info = proj.getInfo()
        crs_str = proj_info.get("crs", "EPSG:32632")
        print(f"   ✅ 使用 Sentinel-2 原生投影: {crs_str}")
        return proj, crs_str
    except Exception as e:
        print(f"   ⚠️ 获取 S2 投影失败: {e}，将使用 UTM 推算")
        return None, None


def get_utm_epsg(lat: float, lon: float) -> str:
    """根据经纬度推算 UTM 投影 EPSG 码"""
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + utm_zone
    else:
        epsg = 32700 + utm_zone
    return f"EPSG:{epsg}"


# ============================================================
# 修改后的 download_facility_aoi 函数（仅改投影获取部分）
# ============================================================
def download_facility_aoi(
    lat: float,
    lon: float,
    facility_name: str,
    year: int,
    output_dir: str,
    aoi_size_meters: int = 1280,
    cloud_thresh: int = 20,
) -> Optional[str]:
    """
    下载单个厂区指定年份的 Sentinel-2 影像
    若该年份无可用影像，返回 None
    """
    half_size = aoi_size_meters / 2
    aoi = ee.Geometry.Point([lon, lat]).buffer(half_size).bounds()

    start_date = f"{year}-06-01"
    end_date = f"{year}-09-30"

    # 获取投影（可能返回 None）
    s2_proj, crs_str = get_s2_projection(aoi, start_date, end_date, cloud_thresh)
    if crs_str is None:
        # 无可用影像，直接回退到 UTM 推算
        crs_str = get_utm_epsg(lat, lon)
        print(f"   ℹ️  使用 UTM 推算投影: {crs_str}")

    # 生成影像（如果无 S2 影像，这一步会失败，需要捕获）
    try:
        rgb = get_sentinel2_rgb(aoi, start_date, end_date, cloud_thresh)
    except RuntimeError as e:
        print(f"   ❌ 获取影像失败: {e}")
        return None

    safe_name = facility_name.replace("/", "_").replace(" ", "_").replace("，", "")
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{safe_name}_{year}.png")

    if os.path.exists(image_path):
        print(f"   ⏭️  已存在，跳过: {image_path}")
        return image_path

    print(f"   ⬇️  下载影像 -> {os.path.basename(image_path)}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tif = os.path.join(tmpdir, "img.tif")
        try:
            geemap.ee_export_image(
                rgb,
                filename=tmp_tif,
                scale=10,
                region=aoi,
                crs=crs_str,
                file_per_band=False
            )
        except Exception as e:
            print(f"   ❌ GEE 导出失败: {e}")
            return None

        # 转换为 PNG
        with rasterio.open(tmp_tif) as src:
            arr = src.read()
            arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        PILImage.fromarray(arr, mode="RGB").save(image_path)

    print(f"   ✅ 完成: {os.path.basename(image_path)}")
    return image_path

# ============================================================
# 3. Sentinel-2 影像下载（移植自原始管线 export_training_patch）
# ============================================================
def get_sentinel2_rgb(
        aoi: ee.Geometry,
        start_date: str,
        end_date: str,
        cloud_thresh: int = 20
) -> ee.Image:
    """
    获取云掩膜后的中值合成 RGB 影像（0-255 uint8）
    原始管线已验证
    """

    def mask_clouds(img):
        qa = img.select("QA60")
        cloud_bit = 1 << 10
        cirrus_bit = 1 << 11
        mask = (qa.bitwiseAnd(cloud_bit).eq(0)).And(qa.bitwiseAnd(cirrus_bit).eq(0))
        return img.updateMask(mask)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .map(mask_clouds)
    )

    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError(
            f"无可用影像（云量 < {cloud_thresh}%），请检查时间范围或降低云量阈值"
        )

    composite = collection.median().clip(aoi)
    rgb = composite.select(["B4", "B3", "B2"])
    rgb_uint8 = rgb.unitScale(0, 3000).multiply(255).clamp(0, 255).uint8()
    return rgb_uint8.rename(["R", "G", "B"])


def download_facility_aoi(
    lat: float,
    lon: float,
    facility_name: str,
    year: int,
    output_dir: str,
    aoi_size_meters: int = 1280,
    cloud_thresh: int = 20,
) -> Optional[str]:
    """
    下载单个厂区指定年份的 Sentinel-2 影像
    若该年份无可用影像，返回 None
    """
    half_size = aoi_size_meters / 2
    aoi = ee.Geometry.Point([lon, lat]).buffer(half_size).bounds()

    start_date = f"{year}-06-01"
    end_date = f"{year}-09-30"

    # 获取投影（可能返回 None）
    s2_proj, crs_str = get_s2_projection(aoi, start_date, end_date, cloud_thresh)
    if crs_str is None:
        crs_str = get_utm_epsg(lat, lon)
        print(f"   ℹ️  使用 UTM 推算投影: {crs_str}")

    # 生成影像——如果无可用影像，捕获异常并返回 None
    try:
        rgb = get_sentinel2_rgb(aoi, start_date, end_date, cloud_thresh)
    except RuntimeError as e:
        print(f"   ⚠️  {year}年无可用影像（云量 < {cloud_thresh}%），已跳过")
        return None

    safe_name = facility_name.replace("/", "_").replace(" ", "_").replace("，", "")
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{safe_name}_{year}.png")

    if os.path.exists(image_path):
        print(f"   ⏭️  已存在，跳过: {image_path}")
        return image_path

    print(f"   ⬇️  下载影像 -> {os.path.basename(image_path)}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tif = os.path.join(tmpdir, "img.tif")
        try:
            geemap.ee_export_image(
                rgb,
                filename=tmp_tif,
                scale=10,
                region=aoi,
                crs=crs_str,
                file_per_band=False
            )
        except Exception as e:
            print(f"   ❌ GEE 导出失败: {e}")
            return None

        with rasterio.open(tmp_tif) as src:
            arr = src.read()
            arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        PILImage.fromarray(arr, mode="RGB").save(image_path)

    print(f"   ✅ 完成: {os.path.basename(image_path)}")
    return image_path


# ============================================================
# 4. 主函数：批量处理
# ============================================================
def parse_year_from_period(period_str: str) -> int:
    """从 "2019-2021" 提取中间年份（作为代表年）"""
    if pd.isna(period_str) or period_str == "":
        return 2020
    parts = period_str.split("-")
    if len(parts) == 2:
        try:
            start = int(parts[0])
            end = int(parts[1])
            return (start + end) // 2
        except ValueError:
            return 2020
    return 2020


def main():
    cfg = CONFIG

    # 初始化 GEE
    init_gee(cfg["project_id"])

    # 读取厂区列表
    if not os.path.exists(cfg["facility_csv"]):
        print(f"❌ 找不到文件: {cfg['facility_csv']}")
        print("请先运行 step1_parse_facility_csv.py 生成 cleaned_facility_list.csv")
        sys.exit(1)

    df = pd.read_csv(cfg["facility_csv"])
    print(f"\n📋 加载厂区列表: {len(df)} 个")

    # 输出目录（image 子目录）
    image_dir = os.path.join(cfg["output_dir"], "image")
    os.makedirs(image_dir, exist_ok=True)
    print(f"📁 输出目录: {image_dir}\n")

    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        company = row.get("company", "unknown")
        facility_name = row.get("facility_name", f"facility_{idx}")
        lat = row.get("lat")
        lon = row.get("lon")

        if pd.isna(lat) or pd.isna(lon):
            print(f"[{idx + 1}/{len(df)}] ⚠️ {facility_name} 坐标缺失，跳过")
            fail_count += 1
            continue

        # 提取基准年和检测年
        baseline_period = row.get("suggested_baseline_period", "2019-2021")
        detection_period = row.get("suggested_detection_period", "2022-2025")

        t0_year = parse_year_from_period(baseline_period)
        t1_year = parse_year_from_period(detection_period)

        print(f"\n[{idx + 1}/{len(df)}] {company} - {facility_name}")
        print(f"   坐标: ({lat:.5f}, {lon:.5f})")
        print(f"   基准年: {t0_year} (来自 {baseline_period})")
        print(f"   检测年: {t1_year} (来自 {detection_period})")

        # 构造带公司前缀的文件夹名（避免不同公司同名厂区冲突）
        safe_prefix = company.replace(" ", "_").replace("/", "_").replace(",", "")
        full_name = f"{safe_prefix}_{facility_name}"

        # 下载 T0
        t0_path = download_facility_aoi(
            lat=float(lat),
            lon=float(lon),
            facility_name=full_name,
            year=t0_year,
            output_dir=image_dir,
            aoi_size_meters=cfg["aoi_size_meters"],
            cloud_thresh=cfg["cloud_thresh"],
        )
        time.sleep(cfg["request_interval"])

        # 下载 T1
        t1_path = download_facility_aoi(
            lat=float(lat),
            lon=float(lon),
            facility_name=full_name,
            year=t1_year,
            output_dir=image_dir,
            aoi_size_meters=cfg["aoi_size_meters"],
            cloud_thresh=cfg["cloud_thresh"],
        )
        time.sleep(cfg["request_interval"])

        if t0_path and t1_path:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 70)
    print("📊 批量下载完成")
    print("=" * 70)
    print(f"  成功: {success_count} 个厂区（两期完整）")
    print(f"  失败: {fail_count} 个厂区")
    print(f"  影像保存在: {image_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()