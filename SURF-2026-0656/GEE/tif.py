"""
export_tif_only.py
==================================================
独立脚本：仅导出活动强度特征图 (.tif) 到 Google Drive
适用于已有训练 PNG，无需重跑切片循环的场景。
==================================================
"""

import ee
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. 配置（请务必修改为你自己的 GEE Project ID）
# ============================================================
PROJECT_ID = "silent-elevator-500103-j4"   # ← 改成你的 GEE Cloud Project ID

# 研究区域（鲁尔区，与你之前下载 240 张图的范围完全一致）
WEST, SOUTH, EAST, NORTH = 6.60, 51.10, 7.60, 51.70
START_DATE = "2023-05-01"
END_DATE   = "2023-09-30"

# 导出配置
EXPORT_FOLDER = "GEE_exports"    # 存到你 Google Drive 的这个文件夹
EXPORT_SCALE  = 30               # 建议 30 米（匹配 Landsat 地温分辨率，切勿用 10）
# ============================================================


# ============================================================
# 1. GEE 初始化
# ============================================================
def init_gee(project_id: str):
    """初始化 Earth Engine"""
    try:
        ee.Initialize(project=project_id)
        print(f"✅ GEE 已初始化, project={project_id}")
    except Exception:
        print("🔑 尚未授权，正在打开浏览器进行登录授权...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print(f"✅ GEE 授权完成并已初始化, project={project_id}")


# ============================================================
# 2. 从原始脚本复制的核心函数（完整保留，无改动）
# ============================================================
def get_sentinel2_for_ndvi(aoi: ee.Geometry, start_date: str, end_date: str,
                            cloud_thresh: int = 20) -> ee.Image:
    """获取用于计算 NDVI 的 Sentinel-2 反射率影像"""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
    )
    return collection.median().clip(aoi)


def compute_ndvi(s2_image: ee.Image) -> ee.Image:
    """NDVI = (NIR - Red) / (NIR + Red)"""
    return s2_image.normalizedDifference(["B8", "B4"]).rename("NDVI")


def get_nighttime_lights(aoi: ee.Geometry,
                          start_date: str, end_date: str) -> ee.Image:
    """VIIRS 夜间灯光月度合成 (avg_rad)"""
    collection = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select("avg_rad")
    )
    return collection.median().clip(aoi).rename("nightlight")


def get_thermal(aoi: ee.Geometry, start_date: str, end_date: str,
                 cloud_thresh: int = 20) -> ee.Image:
    """Landsat 9 地表温度（摄氏度），Collection 2 Level 2"""
    def apply_scale(img):
        thermal = img.select("ST_B10").multiply(0.00341802).add(149.0)
        celsius = thermal.subtract(273.15).rename("temperature_C")
        return img.addBands(celsius, overwrite=True)

    collection = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_thresh))
        .map(apply_scale)
    )
    return collection.select("temperature_C").median().clip(aoi)


def get_s2_projection(aoi: ee.Geometry, start_date: str, end_date: str,
                       cloud_thresh: int = 20) -> ee.Projection:
    """获取该 AOI 对应的 Sentinel-2 原生 UTM 投影"""
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .first()
    )
    return s2.select("B2").projection()


def build_activity_feature_stack(aoi: ee.Geometry,
                                  start_date: str, end_date: str) -> ee.Image:
    """
    把 NDVI / 夜间灯光 / 地表温度 重采样对齐到 Sentinel-2 网格，
    堆叠成多波段影像。波段顺序: [NDVI, nightlight, temperature_C]
    """
    s2_raw      = get_sentinel2_for_ndvi(aoi, start_date, end_date)
    ndvi        = compute_ndvi(s2_raw)
    target_proj = s2_raw.select("B4").projection()

    nightlight = (
        get_nighttime_lights(aoi, start_date, end_date)
        .resample("bilinear")
        .reproject(crs=target_proj, scale=EXPORT_SCALE)  # 用 30m
    )
    thermal = (
        get_thermal(aoi, start_date, end_date)
        .resample("bilinear")
        .reproject(crs=target_proj, scale=EXPORT_SCALE)  # 用 30m
    )
    return ndvi.addBands(nightlight).addBands(thermal).toFloat()

# ============================================================
# 3. 导出主程序
# ============================================================
def main():
    # 初始化
    init_gee(PROJECT_ID)

    # 定义 AOI
    aoi = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH])
    print(f"📍 研究区域: ({WEST:.2f}, {SOUTH:.2f}) -> ({EAST:.2f}, {NORTH:.2f})")
    print(f"📅 时间范围: {START_DATE} 至 {END_DATE}")
    print(f"📏 导出分辨率: {EXPORT_SCALE} 米/像素")

    # 获取 S2 投影信息（用于显式锁定 CRS，防止导出时投影漂移）
    print("🔍 正在获取 Sentinel-2 原生投影...")
    s2_proj = get_s2_projection(aoi, START_DATE, END_DATE)
    proj_info = s2_proj.getInfo()
    crs_str = proj_info.get("crs", "EPSG:32632")
    print(f"   ✅ 投影 CRS: {crs_str}")

    # 构建特征堆栈（服务端计算）
    print("⚙️  正在构建活动特征堆栈 (NDVI + 夜间灯光 + 地表温度)...")
    feature_stack = build_activity_feature_stack(aoi, START_DATE, END_DATE)

    # 提交导出任务到 Google Drive
    print("🚀 正在提交导出任务到 Google Drive...")
    task = ee.batch.Export.image.toDrive(
        image=feature_stack,
        description="Ruhr_Activity_Features_30m",   # 任务名称（在 GEE Tasks 里看到）
        folder=EXPORT_FOLDER,                       # Google Drive 目标文件夹
        region=aoi,
        scale=EXPORT_SCALE,
        crs=crs_str,                               # 显式锁定投影，与 S2 一致
        maxPixels=1e13,                            # 允许大区域导出
        fileFormat="GeoTIFF"
    )
    task.start()

    print("\n" + "="*50)
    print("✅ 导出任务已成功提交！")
    print(f"📋 任务名称: Ruhr_Activity_Features_30m")
    print(f"📁 存储位置: 你的 Google Drive → {EXPORT_FOLDER} 文件夹")
    print(f"🌐 请打开 GEE Tasks 页面查看进度:")
    print(f"   https://code.earthengine.google.com/tasks")
    print("="*50)
    print("\n⏳ 等待任务完成后，去 Google Drive 下载 .tif 文件即可。")

    return task


if __name__ == "__main__":
    main()
