import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')



array = [22]



def apply_svd(image, k):
    """
    应用SVD进行图像去噪和压缩（基于mammography_classification中的实现）
    image: numpy数组（uint8）范围[0, 255]
    k: 保留的奇异值数量
    返回: numpy数组（float32）范围[0, 255]
    """
    try:
        # 转换为float32以增强数值稳定性
        gray = image.astype(np.float32)

        # 归一化图像到[0, 1]范围
        min_val = np.min(gray)
        max_val = np.max(gray)

        # 避免除零错误
        if max_val == min_val:
            return gray

        normalized = (gray - min_val) / (max_val - min_val + 1e-8)

        # 执行SVD分解
        U, S, Vt = np.linalg.svd(normalized, full_matrices=False)

        # 截取前k个奇异值
        k = min(k, len(S))
        S_trunc = np.diag(S[:k])
        reconstructed = U[:, :k] @ S_trunc @ Vt[:k, :]

        # 反归一化
        reconstructed = reconstructed * (max_val - min_val) + min_val
        reconstructed = np.clip(reconstructed, 0, 255)

        return reconstructed.astype(np.float32)
    except Exception as e:
        print(f"SVD处理错误: {e}")
        return image.astype(np.float32)

for k in array:
    # 读取图像
    image_path = "D:/desktop/input/Image_1.jpg" # 输入图片路径
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 转换为RGB格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 增强对比度 (CLAHE)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 高斯模糊
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # 转换为灰度图用于SVD
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)

    # 应用SVD处理
    denoised_gray = apply_svd(gray, k)

    # 将处理后的灰度图转回RGB
    svd_image = cv2.cvtColor(denoised_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # 显示原始图像和处理后的图像
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(blurred)
    plt.title('Blurred Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(svd_image)
    plt.title(f'SVD Denoised (k={k})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存处理后的图像
    output_path = f"D:/desktop/1/Image_1(k={k}).jpg"
    cv2.imwrite(output_path, cv2.cvtColor(svd_image, cv2.COLOR_RGB2BGR))
    print(f"处理后的图像已保存至: {output_path}")


plt.figure(figsize = (15,15))

