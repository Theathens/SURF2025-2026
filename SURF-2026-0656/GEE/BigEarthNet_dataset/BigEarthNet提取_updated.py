import os
import random
import tempfile
import time
import numpy as np
import rasterio
from PIL import Image
import ee
import geemap
from requests.exceptions import ConnectionError, ReadTimeout
from google.api_core.exceptions import DeadlineExceeded, ServiceUnavailable

SCALE = 10  # 空间分辨率（米）
RETRY_LIMIT = 5  # 最大重试次数
RETRY_DELAY_BASE = 5  # 重试初始等待秒数
BATCH_SIZE = 50  # 每批处理数量（可修改）
#SLEEP_BETWEEN_BATCHES =  5 # 每批次处理完休息秒数，避免被限流 --> 改为随机数，见main


def init_gee(project_id):
    """初始化 GEE（自动处理认证）"""
    try:
        ee.Initialize(project=project_id)
        print(f"✅  GEE 已初始化 (project={project_id})")
    except Exception:
        print("🔑 首次认证，请按提示操作...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("✅  GEE 认证完成并初始化")


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

        # 准备输出路径，新增_updated后缀
        fname_base = f"patch_{idx:04d}_updated.png"
        out_image = os.path.join(out_image_dir, fname_base)
        out_mask = os.path.join(out_mask_dir, fname_base)

        # 若文件已存在，跳过下载（断点续传）
        if os.path.exists(out_image) and os.path.exists(out_mask):
            print(f"  跳过已存在文件：{fname_base}")
            return True

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
                img_arr = src.read()  # (3, H, W)
                img_arr = np.transpose(img_arr, (1, 2, 0))
                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
            Image.fromarray(img_arr, mode="RGB").save(out_image)

            with rasterio.open(tmp_mask_tif) as src:
                mask_arr = src.read(1)  # (H, W)
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

    except (ConnectionError, ReadTimeout, DeadlineExceeded, ServiceUnavailable) as net_err:
        raise net_err  # 上层捕获重试
    except Exception as e:
        print(f"  ❌  导出失败: {e}")
        return False


def safe_ee_call(func, *args, **kwargs):
    """
    智能重试包装函数，捕获网络异常，重试等待
    """
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, ReadTimeout, DeadlineExceeded, ServiceUnavailable) as e:
            wait_time = RETRY_DELAY_BASE * attempt
            print(f"⚠️ 网络异常，重试 {attempt}/{RETRY_LIMIT}，等待 {wait_time} 秒...")
            time.sleep(wait_time)
        except Exception as e:
            # 非网络异常，直接抛
            raise e
    raise RuntimeError(f"超过最大重试次数 {RETRY_LIMIT}，操作失败。")


def main():
    # ==================== 配置参数 ====================
    PROJECT_ID = "silent-elevator-500103-j4"  # 你的 GEE Cloud Project ID
    OUTPUT_DIR = "./bigearthnet_output_2000"  # 输出根目录
    NUM_SAMPLES = 5000  # 总采样数量
    # =================================================

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_dir = os.path.join(OUTPUT_DIR, "image")
    mask_dir = os.path.join(OUTPUT_DIR, "mask")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 初始化 GEE
    init_gee(PROJECT_ID)

    # 1. BigEarthNet 图像集合，筛选工业斑块
    print("📡 加载 BigEarthNet 并筛选工业斑块...")
    bigearthnet = ee.ImageCollection("TUBerlin/BigEarthNet/v1")
    industrial = bigearthnet.filter(
        ee.Filter.listContains("labels", "Industrial or commercial units")
    )

    # 获取全部 ID 列表（一次性拉取，12895 个没问题）
    print("📡 正在获取所有工业斑块 ID 列表...")
    all_ids = industrial.aggregate_array("system:id").getInfo()
    total = len(all_ids)
    print(f"✅ 共找到 {total} 个工业斑块")

    if NUM_SAMPLES > total:
        NUM_SAMPLES = total
        print(f"⚠️ 采样数超过总数，改为 {NUM_SAMPLES}")

    # 2. 随机采样
    print(f"🎲 随机采样 {NUM_SAMPLES} 个斑块...")
    sample_ids = random.sample(all_ids, NUM_SAMPLES)

    # 3. 分批处理
    batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"🚀 总共 {batches} 批，每批最多 {BATCH_SIZE} 个斑块")

    idx_global = 0
    failed = []

    for batch_i in range(batches):
        start_idx = batch_i * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, NUM_SAMPLES)
        batch_ids = sample_ids[start_idx:end_idx]
        print(f"\n🗂️ 处理第 {batch_i + 1} 批: 斑块 {start_idx + 1} ~ {end_idx} 共 {len(batch_ids)} 个")

        for offset, img_id in enumerate(batch_ids):
            idx_global = start_idx + offset + 1
            print(f"\n[{idx_global}/{NUM_SAMPLES}] 导出 {img_id} ...")
            img = ee.Image(img_id)

            # 智能重试导出
            success = False
            for attempt in range(1, RETRY_LIMIT + 1):
                try:
                    success = export_single_patch(img, idx_global, image_dir, mask_dir)
                    if success:
                        break
                except (ConnectionError, ReadTimeout, DeadlineExceeded, ServiceUnavailable) as e:
                    wait_time = RETRY_DELAY_BASE * attempt
                    print(f"⚠️ 导出网络异常，重试 {attempt}/{RETRY_LIMIT}，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                except Exception as e:
                    print(f"  错误: {e}")
                    break
            if not success:
                failed.append(idx_global)

        # 批次间休息，避免触发限流
        # 批次间休息，随机 3~10 秒，避免触发限流
        sleep_time = random.randint(3, 10)
        print(f"\n🛑 第 {batch_i + 1} 批处理完毕，休息 {sleep_time} 秒...")
        time.sleep(sleep_time)

    # 4. 总结
    print(f"\n✅ 全部完成！成功导出 {NUM_SAMPLES - len(failed)} 对，失败 {len(failed)} 对")
    if failed:
        print(f"⚠️ 失败的序号: {failed}")
    print(f"📁 图像保存在: {image_dir}")
    print(f"📁 掩码保存在: {mask_dir}")


if __name__ == "__main__":
    main()