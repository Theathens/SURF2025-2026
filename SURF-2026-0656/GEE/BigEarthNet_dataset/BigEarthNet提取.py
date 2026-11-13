"""
export_bigearthnet.py
从 BigEarthNet 导出随机采样的工业斑块 (RGB + 工业掩码)
与现有训练数据格式完全兼容（patch_{idx:04d}.png）
"""

import os
import random
import tempfile
import numpy as np
import rasterio
from PIL import Image
import ee
import geemap

SCALE = 10  # 空间分辨率（米）

def init_gee(project_id):
    """初始化 GEE（自动处理认证）"""
    try:
        ee.Initialize(project=project_id)
        print(f"✅ GEE 已初始化 (project={project_id})")
    except Exception:
        print("🔑 首次认证，请按提示操作...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("✅ GEE 认证完成并初始化")

def get_s2_projection_from_image(img):
    """从 BigEarthNet 影像中获取其原生投影（用于对齐）"""
    return img.select("B2").projection()

def get_corine_industrial_mask(aoi, s2_proj, year=2018):
    """
    用 CORINE 数据生成工业掩码，并重采样到指定投影。
    """
    corine = ee.Image(f"COPERNICUS/CORINE/V20/100m/{year}").clip(aoi)
    mask = corine.eq(121).rename("label").uint8()
    return mask.reproject(crs=s2_proj, scale=SCALE)

def export_single_patch(img, idx, out_image_dir, out_mask_dir):
    """
    导出单个 BigEarthNet 斑块（RGB + 掩码）
    返回 True 表示成功，False 表示失败
    """
    try:
        # 获取斑块几何和投影
        geometry = img.geometry()
        s2_proj = get_s2_projection_from_image(img)
        proj_info = s2_proj.getInfo()
        crs_str = proj_info.get("crs", "EPSG:32632")
        print(f"  投影: {crs_str}")

        # 提取 RGB 波段 (B4=Red, B3=Green, B2=Blue)
        rgb = img.select(["B4", "B3", "B2"])
        # BigEarthNet 反射率已归一化到 0~1，乘 255 并转换为 uint8
        rgb_uint8 = rgb.unitScale(0, 10000).multiply(255).clamp(0, 255).uint8().rename(["R", "G", "B"])
        # 生成工业掩码（使用 CORINE）
        mask = get_corine_industrial_mask(geometry, s2_proj)

        # 准备输出路径
        fname = f"patch_{idx:04d}.png"
        out_image = os.path.join(out_image_dir, fname)
        out_mask = os.path.join(out_mask_dir, fname)

        # 临时文件夹
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_img_tif = os.path.join(tmpdir, "img.tif")
            tmp_mask_tif = os.path.join(tmpdir, "mask.tif")

            # 下载 RGB（强制指定投影）
            geemap.ee_export_image(
                rgb_uint8,
                filename=tmp_img_tif,
                scale=SCALE,
                region=geometry,
                crs=crs_str,
                file_per_band=False
            )

            # 下载掩码（使用相同投影）
            geemap.ee_export_image(
                mask,
                filename=tmp_mask_tif,
                scale=SCALE,
                region=geometry,
                crs=crs_str,
                file_per_band=False
            )

            # 转换 TIF → PNG
            with rasterio.open(tmp_img_tif) as src:
                img_arr = src.read()                 # (3, H, W)
                img_arr = np.transpose(img_arr, (1, 2, 0))
                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
            Image.fromarray(img_arr, mode="RGB").save(out_image)

            with rasterio.open(tmp_mask_tif) as src:
                mask_arr = src.read(1)               # (H, W)
                mask_arr = np.clip(mask_arr, 0, 1).astype(np.uint8)
            Image.fromarray(mask_arr, mode="L").save(out_mask)

        # 校验尺寸是否一致
        with Image.open(out_image) as img_pil:
            w_img, h_img = img_pil.size
        with Image.open(out_mask) as mask_pil:
            w_mask, h_mask = mask_pil.size
        if (w_img, h_img) == (w_mask, h_mask):
            print(f"  [OK] 尺寸验证通过: {w_img}x{h_img}")
            return True
        else:
            print(f"  [WARN] 尺寸不一致: image={w_img}x{h_img}, mask={w_mask}x{h_mask}")
            os.remove(out_image)
            os.remove(out_mask)
            return False

    except Exception as e:
        print(f"  ❌ 导出失败: {e}")
        return False

def main():
    # ==================== 配置参数 ====================
    PROJECT_ID = "silent-elevator-500103-j4"  # 你的 GEE Cloud Project ID
    OUTPUT_DIR = "./bigearthnet_output_2000"  # 输出根目录
    NUM_SAMPLES = 2000  # 随机采样数量（测试用）

    # =================================================

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_dir = os.path.join(OUTPUT_DIR, "image")
    mask_dir = os.path.join(OUTPUT_DIR, "mask")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 初始化 GEE
    init_gee(PROJECT_ID)

    # 1. 从 BigEarthNet 筛选工业斑块
    print("📡 正在加载 BigEarthNet 并筛选工业斑块...")
    bigearthnet = ee.ImageCollection("TUBerlin/BigEarthNet/v1")
    industrial = bigearthnet.filter(
        ee.Filter.listContains("labels", "Industrial or commercial units")
    )
    total = industrial.size().getInfo()
    print(f"✅ 共找到 {total} 个工业斑块")

    # 2. 获取所有工业影像的 ID 列表（在 Python 端处理）
    print("🔄 正在获取所有斑块 ID ...")
    industrial_ids = industrial.aggregate_array("system:id").getInfo()
    print(f"✅ 成功获取 {len(industrial_ids)} 个 ID")

    # 3. 在 Python 端随机采样
    if NUM_SAMPLES > len(industrial_ids):
        NUM_SAMPLES = len(industrial_ids)
        print(f"⚠️ 采样数超过总数，改为 {NUM_SAMPLES}")
    sample_ids = random.sample(industrial_ids, NUM_SAMPLES)
    print(f"🎲 随机采样 {len(sample_ids)} 个斑块")

    # 4. 逐个导出
    failed = []
    for idx, img_id in enumerate(sample_ids, start=1):
        img = ee.Image(img_id)
        print(f"\n[{idx}/{len(sample_ids)}] 导出 {img_id} ...")
        success = export_single_patch(img, idx, image_dir, mask_dir)
        if not success:
            failed.append(idx)

    # 5. 总结
    print(f"\n✅ 全部完成！成功导出 {len(sample_ids) - len(failed)} 对，失败 {len(failed)} 对")
    if failed:
        print(f"⚠️ 失败的序号: {failed}")
    print(f"📁 图像保存在: {image_dir}")
    print(f"📁 掩码保存在: {mask_dir}")

if __name__ == "__main__":
    main()