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
     ──这是任务卡 Week3-4 里两个并列任务的正确拆分方式：
       "U-Net 邊界檢測" 和 "估算活動強度" 是两个独立步骤，
       不要把 NDVI/夜光/热特征当成 U-Net 的输入通道。

依赖安装:
    pip install earthengine-api geemap osmnx geopandas rasterio pillow

首次运行需要:
    1. 注册 Google Earth Engine 账号: https://earthengine.google.com
    2. 运行 ee.Authenticate() 完成一次性授权（会弹出浏览器登录）
    3. 准备一个 GEE Cloud Project ID（在 GEE 控制台创建）

使用示例见文件底部 __main__ 部分。
─────────────────────────────────────────────────
"""

import os
import json
import warnings
import rasterio
from PIL import Image

import numpy as np

warnings.filterwarnings("ignore")

try:
    import ee
    import geemap
except ImportError:
    raise ImportError(
        "请先安装依赖: pip install earthengine-api geemap"
    )


# ============================================================
# 0. 初始化
# ============================================================
def init_gee(project_id: str):
    """
    初始化 Earth Engine。

    project_id: 你在 GEE 控制台创建的 Cloud Project ID
                （新版 earthengine-api 强制要求传入，否则无法调用 API）
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
def get_aoi_from_bbox(west: float, south: float, east: float, north: float) -> ee.Geometry:
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
        # QA60 是 Sentinel-2 官方云掩膜波段，bit10=云, bit11=卷云
        qa = img.select("QA60")
        cloud_bit = 1 << 10
        cirrus_bit = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
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
    # Sentinel-2 SR 反射率范围大致 0~3000，线性拉伸到 0-255 方便存成 PNG
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
# 3. 活动强度特征：NDVI / 夜间灯光 / 地表温度
# ============================================================
def compute_ndvi(s2_image: ee.Image) -> ee.Image:
    """NDVI = (NIR - Red) / (NIR + Red)，范围 -1~1"""
    ndvi = s2_image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return ndvi


def get_nighttime_lights(aoi: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    """
    VIIRS 夜间灯光月度合成数据（avg_rad 波段, 单位 nW/cm²/sr）。
    原生分辨率约 500m，比 Sentinel-2 粗很多，后续需要重采样对齐。
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
    Landsat 9 地表温度（Collection 2 Level 2 产品，已做大气校正）。
    原生分辨率 30m。ST_B10 需要做官方提供的线性换算才是开尔文温度。
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


def build_activity_feature_stack(aoi: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    """
    把 NDVI / 夜间灯光 / 地表温度 重采样对齐到 Sentinel-2 的 10m 网格，
    叠成一个多波段影像。用于 Week5-6 对 U-Net 检测出的工业设施边界
    做 zonal statistics，计算"活动指数"。

    !! 不要把这个 stack 喂给 U-Net，它是给 T5 的下游特征，不是分割输入 !!
    """
    s2_raw = get_sentinel2_for_ndvi(aoi, start_date, end_date)
    ndvi = compute_ndvi(s2_raw)
    target_proj = s2_raw.select("B4").projection()

    nightlight = (
        get_nighttime_lights(aoi, start_date, end_date)
        .resample("bilinear")
        .reproject(crs=target_proj)
    )
    thermal = (
        get_thermal(aoi, start_date, end_date)
        .resample("bilinear")
        .reproject(crs=target_proj)
    )

    stack = ndvi.addBands(nightlight).addBands(thermal)
    return stack  # 波段: [NDVI, nightlight, temperature_C]


# ============================================================
# 4. 工业用地标签 (a) BigEarthNet/CORINE 方案（仅限欧洲）
# ============================================================
def get_corine_industrial_mask(aoi: ee.Geometry, year: int = 2018) -> ee.Image:
    """
    用 CORINE Land Cover 栅格产品提取"工业或商业用地"类别（code=121），
    生成 0/1 二值掩码。这是 BigEarthNet v2.0 标签的同源数据，
    但通过 GEE 直接取栅格更省事，不需要额外下载 reBEN 数据集。

    仅覆盖欧洲区域，非欧洲 AOI 会返回全 0 掩码。
    """
    clc = ee.Image(f"COPERNICUS/CORINE/V20/100m/{year}").clip(aoi)
    # CORINE code 121 = "Industrial or commercial units"
    industrial_mask = clc.eq(121).rename("label").uint8()
    return industrial_mask


# ============================================================
# 5. 工业用地标签 (b) OpenStreetMap 方案（全球可用）
# ============================================================
def get_osm_industrial_mask(aoi: ee.Geometry, aoi_bbox_latlon: tuple) -> ee.Image:
    """
    用 OpenStreetMap 的 landuse=industrial 矢量面，栅格化成掩码。
    适合任何非欧洲研究区。

    aoi_bbox_latlon: (west, south, east, north)，用于 osmnx 下载范围
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("请先安装: pip install osmnx geopandas")

    west, south, east, north = aoi_bbox_latlon
    bbox_poly = (
        f"POLYGON(({west} {south}, {east} {south}, "
        f"{east} {north}, {west} {north}, {west} {south}))"
    )
    from shapely import wkt
    polygon = wkt.loads(bbox_poly)

    print("正在从 OpenStreetMap 下载工业用地矢量面...")
    gdf = ox.features_from_polygon(polygon, tags={"landuse": "industrial"})
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    print(f"共下载到 {len(gdf)} 个工业用地多边形")

    if len(gdf) == 0:
        print("警告: 该区域未检索到 OSM 工业用地标注，返回全 0 掩码")
        return ee.Image(0).clip(aoi).rename("label").uint8()

    fc = geemap.geopandas_to_ee(gdf.reset_index()[["geometry"]])
    industrial_mask = (
        ee.Image(0)
        .paint(fc, 1)
        .clip(aoi)
        .rename("label")
        .uint8()
    )
    return industrial_mask


def get_combined_industrial_mask(aoi: ee.Geometry, aoi_bbox_latlon: tuple,
                                  use_corine: bool = True, use_osm: bool = True,
                                  corine_year: int = 2018) -> ee.Image:
    """
    合并 CORINE 和 OSM 两路标签（取并集），互相补充覆盖盲区。
    欧洲区域两者都有效；非欧洲区域 CORINE 自动归零，只剩 OSM 生效。
    """
    masks = []
    if use_corine:
        masks.append(get_corine_industrial_mask(aoi, corine_year))
    if use_osm:
        masks.append(get_osm_industrial_mask(aoi, aoi_bbox_latlon))

    if not masks:
        raise ValueError("至少启用一种标签来源 (use_corine 或 use_osm)")

    combined = masks[0]
    for m in masks[1:]:
        combined = combined.Or(m)
    return combined.rename("label").uint8()


# ============================================================
# 6. 导出: 与现有 SegmentationDataset 兼容的本地 PNG patch
# ============================================================
def export_training_patch(aoi: ee.Geometry, start_date: str, end_date: str,
                           out_image_path: str, out_mask_path: str,
                           aoi_bbox_latlon: tuple,
                           scale: int = 10,
                           use_corine: bool = True, use_osm: bool = True):
    """
    导出一对 (RGB图像, 标签掩码)。
    由于 geemap 限制，先导出为临时 .tif，再转为 .png 以兼容训练脚本。
    """
    import tempfile
    import os
    import rasterio
    from PIL import Image

    rgb = get_sentinel2_rgb(aoi, start_date, end_date)
    mask = get_combined_industrial_mask(aoi, aoi_bbox_latlon, use_corine, use_osm)

    # 1. 导出为临时 TIF（因为 geemap 只认 .tif）
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_image = os.path.join(tmpdir, "img.tif")
        tmp_mask = os.path.join(tmpdir, "mask.tif")

        print(f"正在导出临时影像 GeoTIFF → {tmp_image}")
        geemap.ee_export_image(rgb, filename=tmp_image, scale=scale, region=aoi, file_per_band=False)

        print(f"正在导出临时掩码 GeoTIFF → {tmp_mask}")
        geemap.ee_export_image(mask, filename=tmp_mask, scale=scale, region=aoi, file_per_band=False)

        # 2. 读取 TIF 并转为 PNG（兼容你的 SegmentationDataset）
        print(f"正在转换影像为 PNG → {out_image_path}")
        with rasterio.open(tmp_image) as src:
            img_data = src.read()  # shape: (3, H, W)
            # 转成 (H, W, 3) 并确保 uint8
            img_png = np.transpose(img_data, (1, 2, 0)).astype(np.uint8)
            Image.fromarray(img_png).save(out_image_path)

        print(f"正在转换掩码为 PNG → {out_mask_path}")
        with rasterio.open(tmp_mask) as src:
            mask_data = src.read(1)  # 单波段
            mask_png = mask_data.astype(np.uint8)
            Image.fromarray(mask_png).save(out_mask_path)

    print("PNG 导出完成")

def export_to_drive(image: ee.Image, aoi: ee.Geometry, description: str,
                     folder: str = "GEE_exports", scale: int = 10):
    """
    大范围 AOI 用这个：导出任务异步提交到你的 Google Drive，
    不卡在本地进程里等待（GEE 服务端处理完会自动放进 Drive 对应文件夹）。
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
# 7. 导出: 活动强度特征图（多波段 GeoTIFF，给 Week5-6 用）
# ============================================================
def export_activity_features(aoi: ee.Geometry, start_date: str, end_date: str,
                              out_path: str, aoi_bbox_latlon: tuple = None,
                              scale: int = 10):
    """
    导出 [NDVI, nightlight, temperature_C] 三波段 GeoTIFF。
    后续用 rasterio 读取，配合 U-Net 检测出的工业设施边界做 zonal stats。
    """
    stack = build_activity_feature_stack(aoi, start_date, end_date)
    print(f"正在导出活动特征图 → {out_path}")
    geemap.ee_export_image(stack, filename=out_path, scale=scale, region=aoi, file_per_band=False)
    print("导出完成。波段顺序: [NDVI, nightlight, temperature_C]")


# ============================================================
# 主函数 / 使用示例
# ============================================================
def main():
    # ==================== 配置 ====================
    PROJECT_ID = "silent-elevator-500103-j4"      # 你的 GEE Cloud Project ID

    # 大范围 AOI（覆盖整个工业区及周边）
    # WEST, SOUTH, EAST, NORTH = 7.0, 51.3, 7.6, 51.6 #台湾高雄工业带
    WEST, SOUTH, EAST, NORTH = 6.70, 51.35, 7.80, 51.65 #德国鲁尔工业带

    START_DATE = "2023-05-01"
    END_DATE = "2023-09-30"

    # 输出根目录（在此目录下自动创建 image/ 和 mask/ 子文件夹）
    OUT_DIR = "./gee_output"

    # 网格边长（度），约 1km（0.01° ≈ 1.1km）
    GRID_SIZE = 0.05   # 可调

    # 是否使用 OSM 和 CORINE（非欧洲区域 CORINE 自动无效）
    USE_CORINE = True
    USE_OSM    = False
    # =============================================

    os.makedirs(OUT_DIR, exist_ok=True)
    image_dir = os.path.join(OUT_DIR, "image")
    mask_dir  = os.path.join(OUT_DIR, "mask")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    init_gee(PROJECT_ID)

    # 生成网格坐标列表
    x_coords = np.arange(WEST, EAST, GRID_SIZE)
    y_coords = np.arange(SOUTH, NORTH, GRID_SIZE)
    patches = []
    for x in x_coords:
        for y in y_coords:
            # 确保网格不超出大范围
            if x + GRID_SIZE <= EAST and y + GRID_SIZE <= NORTH:
                patches.append((x, y, x + GRID_SIZE, y + GRID_SIZE))

    print(f"总共生成 {len(patches)} 个 patch")

    # 循环导出每个 patch
    for idx, (w, s, e, n) in enumerate(patches, start=1):
        # 构建文件名（4位数字编号）
        fname = f"patch_{idx:04d}.png"
        img_path = os.path.join(image_dir, fname)
        msk_path = os.path.join(mask_dir, fname)

        # 定义当前子区域
        aoi = get_aoi_from_bbox(w, s, e, n)
        aoi_bbox = (w, s, e, n)

        print(f"\n处理第 {idx} 个 patch: {w:.4f},{s:.4f} → {e:.4f},{n:.4f}")

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
        except Exception as e:
            print(f"导出第 {idx} 个 patch 失败: {e}")
            continue

    print(f"\n全部完成！图像保存在 {image_dir}，掩码保存在 {mask_dir}")
    print("现在可以将这两个文件夹路径用于训练脚本。")

if __name__ == "__main__":
    main()