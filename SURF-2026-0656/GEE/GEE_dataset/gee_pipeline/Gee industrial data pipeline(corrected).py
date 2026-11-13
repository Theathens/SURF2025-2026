"""
gee_industrial_data_pipeline.py
=================================
SURF 2026 | T2 衛星影像分析師 | Week 1-2 交付物：GEE 腳本模板

功能概览
─────────────────────────────────────────────────
本脚本分成两条独立的产出线，对应任务卡里两个不同阶段的需求：

  ① U-Net 训练用的「图像-掩码」patch（Week 3-4 用）
     - 影像: Sentinel-2 表面反射率 RGB（10m），与你现有
       SegmentationDataset 完全兼容（同样是 3 通道 PNG）
     - 掩码: 工业设施边界（0=背景, 1=工业用地）
       支持两种标签来源，二选一或叠加使用：
         (a) BigEarthNet v2.0 / CORINE Land Cover 的
             "Industrial or commercial units" 类（仅覆盖欧洲10国）
         (b) OpenStreetMap landuse=industrial 矢量面，
             栅格化为掩码（全球可用，适合非欧洲研究区）

  ② 活动强度估算用的特征图（Week 5-6 用，给 T5 算"活动指数"）
     - NDVI（植被指数，从 Sentinel-2 红光/近红外波段计算）
     - 夜间灯光（VIIRS 月度合成）
     - 地表温度（Landsat 9 热红外波段）
     这三者不会喂进 U-Net，而是在 U-Net 检测出工业设施边界之后，
     对边界范围内做 zonal statistics（区域统计），输出活动指数。

修复记录（相对于上一版）
─────────────────────────────────────────────────
[核心修复] image 和 mask 导出尺寸不一致问题（如 557x558 vs 374x572）:
  根本原因:
    ee.Image(0).paint(fc, 1) 和 CORINE 栅格的默认投影不是
    Sentinel-2 的 UTM 网格，导出时行列数与 RGB 影像不同。
  修复方案:
    1. 新增 get_s2_projection() — 从该 AOI 的真实 S2 影像
       中读取原生 UTM 投影，作为所有数据对齐的基准
    2. get_corine_industrial_mask() 新增 s2_proj 参数，
       末尾加 .reproject(crs=s2_proj, scale=10)
    3. get_osm_industrial_mask() 同上
    4. get_combined_industrial_mask() 把 s2_proj 传给子函数
    5. export_training_patch() 先拿 s2_proj，再传入掩码函数，
       保证 RGB 和掩码用完全相同的像素网格导出
    6. 新增 verify_size_match() — 每对 patch 导出后自动检验
       尺寸，不一致时记录编号，全批完成后统一报告

依赖安装:
    pip install earthengine-api geemap osmnx geopandas rasterio pillow
─────────────────────────────────────────────────
"""

import os
import warnings
import tempfile

import numpy as np
import rasterio
from PIL import Image as PILImage

warnings.filterwarnings("ignore")

try:
    import ee
    import geemap
except ImportError:
    raise ImportError("请先安装依赖: pip install earthengine-api geemap")


# ============================================================
# 0. 初始化
# ============================================================
def init_gee(project_id: str):
    """
    初始化 Earth Engine。
    project_id: 你在 GEE 控制台创建的 Cloud Project ID
    """
    try:
        ee.Initialize(project=project_id)
        print(f"GEE 已初始化, project={project_id}")
    except Exception:
        print("尚未授权，正在打开浏览器进行一次性登录授权...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print(f"GEE 授权完成并已初始化, project={project_id}")


# ============================================================
# 1. AOI（研究区域）定义
# ============================================================
def get_aoi_from_bbox(west: float, south: float,
                       east: float, north: float) -> ee.Geometry:
    """用经纬度边界框定义研究区域"""
    return ee.Geometry.Rectangle([west, south, east, north])


def get_aoi_from_place(place_name: str) -> ee.Geometry:
    """
    用地名定义研究区域（内部用 geemap 调 Nominatim 地理编码）。
    例如: get_aoi_from_place("Kaohsiung Industrial Park, Taiwan")
    """
    gdf = geemap.osm_gdf_from_geocode(place_name)
    geom = geemap.geopandas_to_ee(gdf)
    return geom.geometry()


# ============================================================
# 2. Sentinel-2 RGB 合成影像 + 云掩膜
# ============================================================
def get_sentinel2_rgb(aoi: ee.Geometry, start_date: str, end_date: str,
                       cloud_thresh: int = 20) -> ee.Image:
    """
    获取指定时间范围内云量低于 cloud_thresh% 的 Sentinel-2 影像，
    做中位数合成（median composite）以消除残余云/阴影噪声。

    返回: 3 通道 RGB 影像（B4=Red, B3=Green, B2=Blue），10m 分辨率，
          数值范围已拉伸到 0-255（uint8），可直接当作 U-Net 输入。
    """
    def mask_clouds(img):
        qa = img.select("QA60")
        cloud_bit  = 1 << 10
        cirrus_bit = 1 << 11
        mask = (qa.bitwiseAnd(cloud_bit).eq(0)
                .And(qa.bitwiseAnd(cirrus_bit).eq(0)))
        return img.updateMask(mask)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .map(mask_clouds)
    )
    composite = collection.median().clip(aoi)
    rgb = composite.select(["B4", "B3", "B2"])
    rgb_uint8 = rgb.unitScale(0, 3000).multiply(255).clamp(0, 255).uint8()
    return rgb_uint8.rename(["R", "G", "B"])


def get_sentinel2_for_ndvi(aoi: ee.Geometry, start_date: str, end_date: str,
                            cloud_thresh: int = 20) -> ee.Image:
    """获取用于计算 NDVI 的原始反射率影像（不做 0-255 拉伸）"""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
    )
    return collection.median().clip(aoi)


# ============================================================
# [新增] 2.5 获取 Sentinel-2 真实投影（修复尺寸不一致的核心函数）
# ============================================================
def get_s2_projection(aoi: ee.Geometry, start_date: str, end_date: str,
                       cloud_thresh: int = 20) -> ee.Projection:
    """
    从该 AOI 实际存在的 Sentinel-2 影像中读取原生像素投影。

    为什么需要这个函数？
    ────────────────────────────────────────────────────────
    Sentinel-2 按 UTM 分带存储，不同区域对应不同 UTM 带：
      德国鲁尔区 → EPSG:32632（UTM Zone 32N）
      台湾       → EPSG:32651（UTM Zone 51N）

    如果掩码用默认投影（EPSG:4326，地理坐标）导出，
    而影像用 S2 原生 UTM 导出，两者行列数必然不同。

    解决：让影像和掩码都显式 reproject 到同一个投影
    （即 AOI 对应的 S2 UTM 投影 + scale=10m）。
    ────────────────────────────────────────────────────────
    返回: ee.Projection 对象，直接传给 .reproject(crs=..., scale=10)
    """
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .first()
    )
    return s2.select("B2").projection()


# ============================================================
# 3. 活动强度特征：NDVI / 夜间灯光 / 地表温度
# ============================================================
def compute_ndvi(s2_image: ee.Image) -> ee.Image:
    """NDVI = (NIR - Red) / (NIR + Red)，范围 -1~1"""
    return s2_image.normalizedDifference(["B8", "B4"]).rename("NDVI")


def get_nighttime_lights(aoi: ee.Geometry,
                          start_date: str, end_date: str) -> ee.Image:
    """
    VIIRS 夜间灯光月度合成数据（avg_rad 波段, 单位 nW/cm2/sr）。
    原生分辨率约 500m，后续需要重采样对齐到 S2 网格。
    """
    collection = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select("avg_rad")
    )
    return collection.median().clip(aoi).rename("nightlight")


def get_thermal(aoi: ee.Geometry, start_date: str, end_date: str,
                 cloud_thresh: int = 20) -> ee.Image:
    """
    Landsat 9 地表温度（Collection 2 Level 2，已做大气校正）。
    原生分辨率 30m，ST_B10 做官方线性换算转为摄氏度。
    """
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


def build_activity_feature_stack(aoi: ee.Geometry,
                                  start_date: str, end_date: str) -> ee.Image:
    """
    把 NDVI / 夜间灯光 / 地表温度 重采样对齐到 Sentinel-2 的 10m 网格，
    叠成一个多波段影像。用于 Week5-6 对 U-Net 检测出的工业设施边界
    做 zonal statistics，计算"活动指数"。

    !! 不要把这个 stack 喂给 U-Net，它是给 T5 的下游特征，不是分割输入 !!
    """
    s2_raw      = get_sentinel2_for_ndvi(aoi, start_date, end_date)
    ndvi        = compute_ndvi(s2_raw)
    target_proj = s2_raw.select("B4").projection()

    nightlight = (
        get_nighttime_lights(aoi, start_date, end_date)
        .resample("bilinear")
        .reproject(crs=target_proj, scale=10)
    )
    thermal = (
        get_thermal(aoi, start_date, end_date)
        .resample("bilinear")
        .reproject(crs=target_proj, scale=10)
    )
    return ndvi.addBands(nightlight).addBands(thermal)
    # 波段顺序: [NDVI, nightlight, temperature_C]


# ============================================================
# 4. 工业用地标签 (a) CORINE 方案（仅限欧洲）
# ============================================================
def get_corine_industrial_mask(aoi: ee.Geometry,
                                s2_proj: ee.Projection,
                                year: int = 2018) -> ee.Image:
    """
    用 CORINE Land Cover 栅格产品提取"工业或商业用地"类别（code=121），
    生成 0/1 二值掩码。仅覆盖欧洲区域。

    [修复] 新增 s2_proj 参数：
    CORINE 原生分辨率为 100m，坐标系为 EPSG:3035（欧洲等面积投影），
    不 reproject 直接导出会与 Sentinel-2 的 UTM 10m 网格不对齐。
    加了 .reproject(crs=s2_proj, scale=10) 后，GEE 会在服务端
    把 CORINE 重采样到与 RGB 影像完全相同的像素网格再下载。
    """
    clc = ee.Image(f"COPERNICUS/CORINE/V20/100m/{year}").clip(aoi)
    industrial_mask = clc.eq(121).rename("label").uint8()

    # ── 核心修复：强制对齐到 S2 投影 ─────────────────────────────
    return industrial_mask.reproject(crs=s2_proj, scale=10)


# ============================================================
# 5. 工业用地标签 (b) OpenStreetMap 方案（全球可用）
# ============================================================
def get_osm_industrial_mask(aoi: ee.Geometry,
                             aoi_bbox_latlon: tuple,
                             s2_proj: ee.Projection) -> ee.Image:
    """
    用 OpenStreetMap 的 landuse=industrial 矢量面，栅格化成掩码。
    适合任何非欧洲研究区。

    [修复] 新增 s2_proj 参数：
    ee.Image(0).paint(fc, 1) 是矢量转栅格操作，GEE 默认会用
    EPSG:4326（经纬度，1度/像素）作为输出投影，这和 Sentinel-2 的
    UTM 10m 网格完全不同，导出后行列数相差悬殊（就是你看到的
    557x558 vs 374x572 问题的根源）。
    加了 .reproject(crs=s2_proj, scale=10) 后，两者网格完全一致。
    """
    try:
        import osmnx as ox
        from shapely import wkt
    except ImportError:
        raise ImportError("请先安装: pip install osmnx geopandas")

    west, south, east, north = aoi_bbox_latlon
    bbox_poly = (
        f"POLYGON(({west} {south}, {east} {south}, "
        f"{east} {north}, {west} {north}, {west} {south}))"
    )
    polygon = wkt.loads(bbox_poly)

    print("  正在从 OpenStreetMap 下载工业用地矢量面...")
    gdf = ox.features_from_polygon(polygon, tags={"landuse": "industrial"})
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    print(f"  共下载到 {len(gdf)} 个工业用地多边形")

    if len(gdf) == 0:
        print("  警告: 该区域未检索到 OSM 工业用地标注，返回全 0 掩码")
        return (ee.Image(0).clip(aoi).rename("label").uint8()
                .reproject(crs=s2_proj, scale=10))

    fc = geemap.geopandas_to_ee(gdf.reset_index()[["geometry"]])
    industrial_mask = (
        ee.Image(0)
        .paint(fc, 1)
        .clip(aoi)
        .rename("label")
        .uint8()
    )

    # ── 核心修复：强制对齐到 S2 投影 ─────────────────────────────
    return industrial_mask.reproject(crs=s2_proj, scale=10)


def get_combined_industrial_mask(aoi: ee.Geometry,
                                  aoi_bbox_latlon: tuple,
                                  s2_proj: ee.Projection,
                                  use_corine: bool = True,
                                  use_osm: bool = True,
                                  corine_year: int = 2018) -> ee.Image:
    """
    合并 CORINE 和 OSM 两路标签（取并集），互相补充覆盖盲区。
    欧洲区域两者都有效；非欧洲区域 CORINE 自动归零，只剩 OSM 生效。

    [修复] 把 s2_proj 传递给两个子函数，确保两路标签
    都对齐到同一像素网格后再做 Or 合并。
    """
    masks = []
    if use_corine:
        masks.append(get_corine_industrial_mask(aoi, s2_proj, corine_year))
    if use_osm:
        masks.append(get_osm_industrial_mask(aoi, aoi_bbox_latlon, s2_proj))

    if not masks:
        raise ValueError("至少启用一种标签来源 (use_corine 或 use_osm)")

    combined = masks[0]
    for m in masks[1:]:
        combined = combined.Or(m)
    return combined.rename("label").uint8()


# ============================================================
# [新增] 5.5 导出后尺寸一致性验证
# ============================================================
def verify_size_match(image_path: str, mask_path: str) -> bool:
    """
    导出 PNG 后立刻检查 image 和 mask 的像素尺寸是否完全一致。
    返回 True 表示通过，False 表示不一致（供调用方决定是否重试）。
    """
    with PILImage.open(image_path) as img:
        img_size = img.size   # (W, H)
    with PILImage.open(mask_path) as msk:
        msk_size = msk.size

    if img_size == msk_size:
        print(f"  [OK] 尺寸验证通过: {img_size[0]}x{img_size[1]}")
        return True
    else:
        print(
            f"  [WARN] 尺寸不一致！"
            f"image={img_size[0]}x{img_size[1]}, "
            f"mask={msk_size[0]}x{msk_size[1]}"
        )
        return False


# ============================================================
# 6. 导出: 与现有 SegmentationDataset 兼容的本地 PNG patch
# ============================================================
def export_training_patch(aoi: ee.Geometry,
                           start_date: str, end_date: str,
                           out_image_path: str, out_mask_path: str,
                           aoi_bbox_latlon: tuple,
                           scale: int = 10,
                           use_corine: bool = True,
                           use_osm: bool = True):
    """
    导出一对 (RGB图像, 标签掩码)，最终保存为 PNG，
    文件格式与现有 SegmentationDataset 完全兼容。

    [修复 v2] 为什么上一版 .reproject() 没有生效？
    ─────────────────────────────────────────────────────────
    geemap.ee_export_image 内部走的是 GEE thumbnails 下载接口，
    不是正式的 Export.image 接口。thumbnails 接口会忽略服务端
    的懒执行 .reproject() 调用，用自己的默认投影下载数据，
    所以掩码始终以 EPSG:4326 输出，与影像的 UTM 网格不一致。

    正确做法：把 CRS 字符串显式传给 geemap.ee_export_image 的
    crs 参数，让下载接口本身强制使用指定投影，而不是靠服务端 reproject。

    修复流程：
      step1: 获取 S2 投影对象（ee.Projection）
      step2: 调用 .getInfo() 提取 CRS 字符串（如 "EPSG:32632"）
             ← 这是 v2 的关键新增步骤
      step3: 生成 RGB 影像和掩码影像
      step4: ee_export_image 两次调用都传入 crs=crs_str
             ← 这才是让两者像素网格完全对齐的真正有效方法
      step5: TIF → PNG 格式转换
      step6: verify_size_match 验证尺寸
    ─────────────────────────────────────────────────────────
    """
    assert out_image_path.lower().endswith(".png"), "out_image_path 应以 .png 结尾"
    assert out_mask_path.lower().endswith(".png"),  "out_mask_path 应以 .png 结尾"

    # ── step1: 获取 S2 投影对象 ──────────────────────────────────
    print("  正在获取 Sentinel-2 原生投影...")
    s2_proj = get_s2_projection(aoi, start_date, end_date)

    # ── step2: 提取 CRS 字符串（v2 关键新增） ────────────────────
    # ee.Projection.getInfo() 返回形如:
    #   {"type": "SR-ORG:...", "crs": "EPSG:32632", "transform": [...]}
    # 我们只需要 crs 字符串，显式传给下载接口，让两次下载用完全相同的投影。
    proj_info = s2_proj.getInfo()
    crs_str   = proj_info.get("crs", "EPSG:32632")
    print(f"  投影: {crs_str}")

    # ── step3: 生成影像和掩码（服务端 .reproject 保留但不再是主要保障） ──
    rgb  = get_sentinel2_rgb(aoi, start_date, end_date)
    mask = get_combined_industrial_mask(
        aoi=aoi,
        aoi_bbox_latlon=aoi_bbox_latlon,
        s2_proj=s2_proj,
        use_corine=use_corine,
        use_osm=use_osm,
    )

    # ── step4: 下载 TIF，两次都传 crs=crs_str（真正的修复点） ─────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_image = os.path.join(tmpdir, "img.tif")
        tmp_mask  = os.path.join(tmpdir, "mask.tif")

        print("  正在下载 RGB 影像...")
        geemap.ee_export_image(
            rgb, filename=tmp_image,
            scale=scale, region=aoi,
            crs=crs_str,          # ← 显式锁定投影
            file_per_band=False
        )

        print("  正在下载工业用地掩码...")
        geemap.ee_export_image(
            mask, filename=tmp_mask,
            scale=scale, region=aoi,
            crs=crs_str,          # ← 与影像完全相同的投影
            file_per_band=False
        )

        # ── step5: TIF → PNG ─────────────────────────────────────
        with rasterio.open(tmp_image) as src:
            img_arr = src.read()                        # (3, H, W)
            img_arr = np.transpose(img_arr, (1, 2, 0)) # (H, W, 3)
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        PILImage.fromarray(img_arr, mode="RGB").save(out_image_path)

        with rasterio.open(tmp_mask) as src:
            msk_arr = src.read(1)                       # (H, W)
            msk_arr = np.clip(msk_arr, 0, 1).astype(np.uint8)
        PILImage.fromarray(msk_arr, mode="L").save(out_mask_path)

    # ── step6: 验证尺寸 ──────────────────────────────────────────
    verify_size_match(out_image_path, out_mask_path)


# ============================================================
# 7. 大范围 AOI 异步导出到 Google Drive
# ============================================================
def export_to_drive(image: ee.Image, aoi: ee.Geometry,
                     description: str, folder: str = "GEE_exports",
                     scale: int = 10):
    """
    大范围 AOI 用这个：导出任务异步提交到你的 Google Drive，
    不卡在本地进程里等待。
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=aoi,
        scale=scale,
        maxPixels=1e13,
    )
    task.start()
    print(f"导出任务已提交: {description}（在 GEE 控制台 Tasks 标签页查看进度）")
    return task


# ============================================================
# 8. 导出: 活动强度特征图（多波段 GeoTIFF，给 Week5-6 用）
# ============================================================
def export_activity_features(aoi: ee.Geometry,
                              start_date: str, end_date: str,
                              out_path: str,
                              scale: int = 10):
    """
    导出 [NDVI, nightlight, temperature_C] 三波段 GeoTIFF。
    后续用 rasterio 读取，配合 U-Net 检测出的工业设施边界做 zonal stats。
    """
    assert out_path.lower().endswith(".tif"), "活动特征图请用 .tif 后缀"
    stack = build_activity_feature_stack(aoi, start_date, end_date)
    print(f"正在导出活动特征图 -> {out_path}")
    geemap.ee_export_image(
        stack, filename=out_path,
        scale=scale, region=aoi, file_per_band=False
    )
    print("导出完成。波段顺序: [NDVI, nightlight, temperature_C]")


# ============================================================
# 主函数 / 使用示例
# ============================================================
def main():
    # ==================== 配置（按实际情况修改） ====================
    PROJECT_ID = "silent-elevator-500103-j4"   # 你的 GEE Cloud Project ID

    # 研究区域（经纬度，西/南/东/北）
    # 示例：德国鲁尔工业带（欧洲，CORINE 标签有效）
    WEST, SOUTH, EAST, NORTH = 6.60, 51.10, 7.60, 51.70
    # 台湾高雄临海工业区（非欧洲，需改 USE_OSM=True）
    # WEST, SOUTH, EAST, NORTH = 120.28, 22.55, 120.36, 22.62

    START_DATE = "2023-05-01"
    END_DATE   = "2023-09-30"

    # 输出根目录（自动创建 image/ 和 mask/ 子文件夹）
    OUT_DIR = "./gee_output_ruhr_industrial"

    # 网格边长（度），0.05 约 5.5km，导出后约 550x550 px（10m/px）
    GRID_SIZE = 0.05

    # 标签来源（欧洲区域 CORINE 有效，非欧洲改 USE_OSM=True）
    USE_CORINE = True
    USE_OSM    = False
    # ================================================================

    os.makedirs(OUT_DIR, exist_ok=True)
    image_dir = os.path.join(OUT_DIR, "image")
    mask_dir  = os.path.join(OUT_DIR, "mask")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir,  exist_ok=True)

    init_gee(PROJECT_ID)

    # 生成网格坐标列表
    x_coords = np.arange(WEST,  EAST,  GRID_SIZE)
    y_coords = np.arange(SOUTH, NORTH, GRID_SIZE)
    patches  = [
        (x, y, x + GRID_SIZE, y + GRID_SIZE)
        for x in x_coords for y in y_coords
        if x + GRID_SIZE <= EAST and y + GRID_SIZE <= NORTH
    ]
    print(f"总共生成 {len(patches)} 个 patch")

    size_errors = []   # 记录尺寸不一致的 patch 编号

    for idx, (w, s, e, n) in enumerate(patches, start=1):
        fname    = f"patch_{idx:04d}.png"
        img_path = os.path.join(image_dir, fname)
        msk_path = os.path.join(mask_dir,  fname)
        aoi      = get_aoi_from_bbox(w, s, e, n)
        aoi_bbox = (w, s, e, n)

        print(f"\n[{idx}/{len(patches)}] "
              f"({w:.4f},{s:.4f}) -> ({e:.4f},{n:.4f})")

        try:
            export_training_patch(
                aoi=aoi,
                start_date=START_DATE,
                end_date=END_DATE,
                out_image_path=img_path,
                out_mask_path=msk_path,
                aoi_bbox_latlon=aoi_bbox,
                scale=10,
                use_corine=USE_CORINE,
                use_osm=USE_OSM,
            )
        except Exception as ex:
            print(f"  导出失败: {ex}")
            size_errors.append(idx)
            continue

        # 导出后立刻验证尺寸（[新增]）
        if not verify_size_match(img_path, msk_path):
            size_errors.append(idx)

    # 导出活动强度特征图（Week5-6 用）
    print("\n正在导出整体 AOI 的活动强度特征图...")
    full_aoi = get_aoi_from_bbox(WEST, SOUTH, EAST, NORTH)
    try:
        export_activity_features(
            aoi=full_aoi,
            start_date=START_DATE,
            end_date=END_DATE,
            out_path=os.path.join(OUT_DIR, "activity_features.tif"),
            scale=10,
        )
    except Exception as ex:
        print(f"活动特征图导出失败: {ex}")

    print(f"\n全部完成！图像保存在 {image_dir}，掩码保存在 {mask_dir}")
    if size_errors:
        print(f"[WARN] 以下 patch 尺寸不一致，建议删除后重新导出: "
              f"{size_errors}")
    else:
        print("[OK] 所有 patch 尺寸验证通过，可直接用于训练。")
    print("将 image/ 和 mask/ 路径填入训练脚本的 image_dir / mask_dir 即可。")


if __name__ == "__main__":
    main()



    