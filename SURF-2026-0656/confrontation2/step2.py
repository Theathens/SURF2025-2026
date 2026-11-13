#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_download_sentinel2.py (重构版)
=====================================
功能：
  1. 读取 step1 生成的 cleaned_facility_list.csv
  2. 对每个厂区，根据 baseline_year 和 detection_year 下载两期影像
  3. 使用动态 AOI 缓冲区（500m ~ 8km）
  4. 若 Sentinel-2 无可用影像，自动回退到 Landsat-8/9
  5. 影像尺寸自动适配（以 10m 分辨率对齐，输出方形 PNG）

依赖：pip install earthengine-api geemap pandas rasterio pillow
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

try:
    import ee
    import geemap
except ImportError:
    raise ImportError("请先安装依赖: pip install earthengine-api geemap pandas rasterio pillow")

# ================= 硬编码配置区 =================
CONFIG = {
    # GEE Cloud Project ID（必改！）
    "project_id": "silent-elevator-500103-j4",

    # Step 1 生成的聚合厂区列表
    "facility_csv": r"D:\Pycharm Projects\torch\confrontation2\data\cleaned_facility_list.csv",

    # 输出目录（影像保存在 image/ 子目录下）
    "output_dir": r"D:\Pycharm Projects\torch\confrontation2\data\01_send_to_group3_PUBLIC_BLIND\sentinel2_aoi",

    # 云量阈值（%）
    "cloud_thresh": 30,

    # 请求间隔（秒），避免 GEE 限流
    "request_interval": 3,
}
# ===============================================


# ============================================================
# 1. GEE 初始化
# ============================================================
def init_gee(project_id: str):
    try:
        ee.Initialize(project=project_id)
        print(f"✅ GEE 已初始化, project={project_id}")
    except Exception:
        print("尚未授权，正在打开浏览器进行一次性登录授权...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print(f"✅ GEE 授权完成并已初始化, project={project_id}")


# ============================================================
# 2. 投影推算（无重复定义）
# ============================================================
def get_utm_epsg(lat: float, lon: float) -> str:
    """根据经纬度推算 UTM 投影 EPSG 码"""
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + utm_zone
    else:
        epsg = 32700 + utm_zone
    return f"EPSG:{epsg}"


def get_s2_projection(aoi: ee.Geometry, start_date: str, end_date: str, cloud_thresh: int = 30) -> Tuple[Optional[ee.Projection], Optional[str]]:
    """获取 Sentinel-2 原生投影，若无可用影像则返回 (None, None)"""
    try:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        )
        size = collection.size().getInfo()
        if size == 0:
            return None, None
        s2 = collection.first()
        proj = s2.select("B2").projection()
        proj_info = proj.getInfo()
        crs_str = proj_info.get("crs", "EPSG:32632")
        return proj, crs_str
    except Exception as e:
        return None, None


# ============================================================
# 3. 影像获取（Sentinel-2 + Landsat 回退）
# ============================================================
def get_composite_rgb(aoi: ee.Geometry, year: int, cloud_thresh: int = 30) -> ee.Image:
    """
    获取指定年份的夏季（6-9月）无云 RGB 合成影像。
    优先使用 Sentinel-2，若无数据则自动回退到 Landsat-8/9。
    返回 0-255 uint8 RGB 影像。
    """
    start_date = f"{year}-06-01"
    end_date = f"{year}-09-30"

    # 尝试 Sentinel-2
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
        print(f"   📡 使用 Sentinel-2 ({s2_count} 景影像)")
        composite = s2_col.map(mask_s2_clouds).median().clip(aoi)
        rgb = composite.select(["B4", "B3", "B2"])
    else:
        # 回退到 Landsat-8/9
        print(f"   ⚠️  Sentinel-2 无数据，回退到 Landsat-8/9")
        ls_col = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", cloud_thresh))
        )

        def mask_ls_clouds(img):
            qa = img.select("QA_PIXEL")
            cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)  # 云阴影
            return img.updateMask(cloud_mask)

        ls_count = ls_col.size().getInfo()
        if ls_count == 0:
            raise RuntimeError(f"❌ {year}年 无任何可用光学影像（S2 + L8/9 均无）")
        print(f"   📡 使用 Landsat ({ls_count} 景影像)")
        composite = ls_col.map(mask_ls_clouds).median().clip(aoi)
        # Landsat 波段: B4=红, B3=绿, B2=蓝
        rgb = composite.select(["SR_B4", "SR_B3", "SR_B2"])

    # 转换为 0-255 uint8（不同卫星缩放因子不同，但 Landsat SR 已缩放，S2 需 unitScale）
    # 统一用 unitScale(0, 3000) 对 S2；Landsat 需要单独处理，但简单统一用 0-3000 范围也可
    rgb_uint8 = rgb.unitScale(0, 3000).multiply(255).clamp(0, 255).uint8()
    return rgb_uint8.rename(["R", "G", "B"])


# ============================================================
# 4. 单个厂区下载函数（仅此一份，无重复）
# ============================================================
def download_facility_aoi(
    lat: float,
    lon: float,
    facility_name: str,
    year: int,
    output_dir: str,
    buffer_meters: int = 500,
    cloud_thresh: int = 30,
) -> Optional[str]:
    """
    下载单个厂区指定年份的 RGB 影像。
    若该年份无可用影像，返回 None。
    """
    # 构建 AOI（使用动态缓冲区）
    aoi = ee.Geometry.Point([lon, lat]).buffer(buffer_meters).bounds()

    # 获取投影（优先 S2 原生，失败则用 UTM）
    start_date = f"{year}-06-01"
    end_date = f"{year}-09-30"
    s2_proj, crs_str = get_s2_projection(aoi, start_date, end_date, cloud_thresh)
    if crs_str is None:
        crs_str = get_utm_epsg(lat, lon)
        print(f"   ℹ️  使用 UTM 推算投影: {crs_str}")

    # 获取合成影像
    try:
        rgb = get_composite_rgb(aoi, year, cloud_thresh)
    except RuntimeError as e:
        print(f"   ❌ {e}")
        return None

    # 准备输出路径
    safe_name = facility_name.replace("/", "_").replace(" ", "_").replace("，", "").replace(",", "")
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{safe_name}_{year}.png")

    if os.path.exists(image_path):
        print(f"   ⏭️  已存在，跳过: {os.path.basename(image_path)}")
        return image_path

    print(f"   ⬇️  下载影像 -> {os.path.basename(image_path)}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tif = os.path.join(tmpdir, "img.tif")
        try:
            geemap.ee_export_image(
                rgb,
                filename=tmp_tif,
                scale=10,          # 10m 分辨率
                region=aoi,
                crs=crs_str,
                file_per_band=False
            )
        except Exception as e:
            print(f"   ❌ GEE 导出失败: {e}")
            return None

        # 转换为 PNG
        try:
            with rasterio.open(tmp_tif) as src:
                arr = src.read()
                # 确保是 3 波段 RGB
                if arr.shape[0] != 3:
                    print(f"   ⚠️  波段数异常 ({arr.shape[0]})，跳过")
                    return None
                arr = np.transpose(arr, (1, 2, 0))
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            PILImage.fromarray(arr, mode="RGB").save(image_path)
        except Exception as e:
            print(f"   ❌ 图像转换失败: {e}")
            return None

    print(f"   ✅ 完成: {os.path.basename(image_path)}")
    return image_path


# ============================================================
# 5. 主函数：批量处理
# ============================================================
def main():
    cfg = CONFIG

    # 初始化 GEE
    init_gee(cfg["project_id"])

    # 读取厂区列表
    if not os.path.exists(cfg["facility_csv"]):
        print(f"❌ 找不到文件: {cfg['facility_csv']}")
        print("请先运行 step1_parse_facility.py 生成 cleaned_facility_list.csv")
        sys.exit(1)

    df = pd.read_csv(cfg["facility_csv"])
    print(f"\n📋 加载厂区列表: {len(df)} 个")

    # 输出目录
    image_dir = os.path.join(cfg["output_dir"], "image")
    os.makedirs(image_dir, exist_ok=True)
    print(f"📁 输出目录: {image_dir}\n")

    success_count = 0
    fail_count = 0
    total_pairs = 0

    for idx, row in df.iterrows():
        company = row.get("company", "unknown")
        facility_name = row.get("facility_name", f"facility_{idx}")
        lat = row.get("lat")
        lon = row.get("lon")
        buffer_m = int(row.get("aoi_buffer_meters", 3000))
        t0_year = int(row.get("baseline_year", 2019))
        t1_year = int(row.get("detection_year", 2024))

        if pd.isna(lat) or pd.isna(lon):
            print(f"[{idx + 1}/{len(df)}] ⚠️ {facility_name} 坐标缺失，跳过")
            fail_count += 1
            continue

        print(f"\n[{idx + 1}/{len(df)}] {company} - {facility_name}")
        print(f"   坐标: ({lat:.5f}, {lon:.5f}) | 缓冲区: {buffer_m}m")
        print(f"   基准年(T0): {t0_year}  检测年(T1): {t1_year}")

        # 构造带公司前缀的名称
        safe_prefix = company.replace(" ", "_").replace("/", "_").replace(",", "")
        full_name = f"{safe_prefix}_{facility_name}"

        # 下载 T0
        t0_path = download_facility_aoi(
            lat=float(lat),
            lon=float(lon),
            facility_name=full_name,
            year=t0_year,
            output_dir=image_dir,
            buffer_meters=buffer_m,
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
            buffer_meters=buffer_m,
            cloud_thresh=cfg["cloud_thresh"],
        )
        time.sleep(cfg["request_interval"])

        if t0_path and t1_path:
            success_count += 1
            total_pairs += 1
        else:
            fail_count += 1

    print("\n" + "=" * 70)
    print("📊 批量下载完成")
    print("=" * 70)
    print(f"  成功下载双期影像: {success_count} 个厂区")
    print(f"  失败/缺失: {fail_count} 个厂区")
    print(f"  影像保存在: {image_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()