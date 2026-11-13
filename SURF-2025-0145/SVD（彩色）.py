import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

# 检测GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if device.type == "cuda":
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

# 保留奇异值数量配置
SINGULAR_VALUES = [1, 10, 22, 50, 75, 100, 125, 150]


def apply_svd_to_channel(channel_tensor, k):
    """
    对单个通道应用SVD进行图像压缩 (GPU加速)
    channel_tensor: torch.Tensor (float32) 范围[0, 255]
    k: 保留的奇异值数量
    返回: torch.Tensor (float32) 范围[0, 255]
    """
    try:
        # 归一化处理 (在GPU上)
        min_val = torch.min(channel_tensor)
        max_val = torch.max(channel_tensor)

        # 避免除零错误
        if max_val - min_val < 1e-8:
            return channel_tensor

        # 归一化到[0, 1]范围
        normalized = (channel_tensor - min_val) / (max_val - min_val)

        # 执行SVD分解 (使用GPU)
        U, S, V = torch.linalg.svd(normalized)

        # 截断奇异值
        k = min(k, len(S))
        S_trunc = torch.diag(S[:k])

        # 重构图像 (高效矩阵乘法)
        reconstructed = U[:, :k] @ S_trunc @ V[:k, :]

        # 反归一化
        denoised = reconstructed * (max_val - min_val) + min_val

        # 确保值在有效范围内
        return torch.clamp(denoised, 0, 255)

    except Exception as e:
        print(f"SVD处理错误: {e}")
        return channel_tensor


def process_image(image_path, output_dir, singular_values=SINGULAR_VALUES):
    """
    处理单张图像并保存结果
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 读取图像
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 提取文件名
    filename = Path(image_path).stem

    # 转换为RGB并分离通道
    rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    r_orig, g_orig, b_orig = cv2.split(rgb_image)

    # 将通道数据转换为PyTorch张量并移到GPU
    r_tensor = torch.from_numpy(r_orig.astype(np.float32)).to(device)
    g_tensor = torch.from_numpy(g_orig.astype(np.float32)).to(device)
    b_tensor = torch.from_numpy(b_orig.astype(np.float32)).to(device)

    print(f"开始处理图像: {filename} ({rgb_image.shape[1]}x{rgb_image.shape[0]})")
    print(f"将处理 {len(singular_values)} 个k值: {singular_values}")

    # 记录总时间
    total_start = time.time()

    # 对每个奇异值数量进行处理
    for i, k in enumerate(singular_values):
        iter_start = time.time()
        print(f"\n处理 k={k} ({i + 1}/{len(singular_values)})")

        # 对各通道应用SVD (GPU加速)
        r_denoised = apply_svd_to_channel(r_tensor, k)
        g_denoised = apply_svd_to_channel(g_tensor, k)
        b_denoised = apply_svd_to_channel(b_tensor, k)

        # 将结果移回CPU并转换为numpy数组
        r_np = r_denoised.cpu().numpy()
        g_np = g_denoised.cpu().numpy()
        b_np = b_denoised.cpu().numpy()

        # 合并通道
        denoised_image = cv2.merge([r_np, g_np, b_np])
        display_image = denoised_image.astype(np.uint8)

        # 创建对比图
        plt.figure(figsize=(15, 5), dpi=100)

        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(display_image)
        plt.title(f'Compressed Image (k={k})')
        plt.axis('off')

        plt.tight_layout()

        # # 保存对比图
        # comp_path = os.path.join(output_dir, f"{filename}_comp_k={k}.jpg")
        # plt.savefig(comp_path, bbox_inches='tight')
        # plt.close()

        # 保存结果图像 (BGR格式)
        output_path = os.path.join(output_dir, f"{filename}_k={k}.jpg")
        bgr_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_image)

        iter_time = time.time() - iter_start
        print(f"  k={k}: 处理时间 = {iter_time:.2f}秒 | 已保存: {output_path}")

    total_time = time.time() - total_start
    print(f"\n图像处理完成! 总时间: {total_time:.2f}秒")
    print(f"平均每k值处理时间: {total_time / len(singular_values):.2f}秒")

    return total_time


if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "D:/desktop/input"
    OUTPUT_DIR = "D:/desktop/output"

    # 支持多图像处理
    IMAGE_PATHS = [
        os.path.join(INPUT_DIR, "Fenrir.jpg"),
        # 可以添加更多图像路径
        # os.path.join(INPUT_DIR, "image2.jpg"),
    ]

    # 处理所有图像
    total_processing_time = 0
    for img_path in IMAGE_PATHS:
        if os.path.exists(img_path):
            img_time = process_image(img_path, OUTPUT_DIR)
            total_processing_time += img_time
        else:
            print(f"警告: 图像不存在 - {img_path}")

    print(f"\n所有处理完成! 总耗时: {total_processing_time:.2f}秒")

    # 显示GPU内存使用情况
    if device.type == "cuda":
        print("\nGPU内存使用统计:")
        print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"  已保留: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")
        print(f"  最大已分配: {torch.cuda.max_memory_allocated(0) / 1024 ** 2:.2f} MB")