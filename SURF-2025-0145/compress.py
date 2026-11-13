import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# 读取图像
image_path = "D:/desktop/w2/w2/mammography_images/train/Image_1.jpg"   #输入读取图片路径
image = Image.open(image_path)
image = np.array(image)

# 对每个RGB通道进行 SURF-2025-0145 分解和重构
def svd_compression(image, k):
    compressed_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[2]):     #获取图像的颜色通道数，遍历每个颜色通道
        U, Sigma, VT = np.linalg.svd(image[:, :, i])    #提取当前颜色通道的二维矩阵，对该二维矩阵进行 SURF-2025-0145 分解
        compressed_image[:, :, i] = U[:, :k] @ np.diag(Sigma[:k]) @ VT[:k, :]
    return compressed_image.astype(np.uint8)    #将重构后的通道数据存入 compressed_image 的对应通道

# 保留前 k 个奇异值
# k 值越小，压缩比越高，但图像质量会下降；k 值越大，图像质量越高，但压缩比降低
print("k值测试：10，50，100，500，750，1000")
print("请输入k值：")
n = input()
k = int(n)
compressed_image = svd_compression(image, k)


# 显示原始图像和压缩后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(compressed_image)
plt.title(f'Compressed Image (k={k})')
plt.axis('off')

plt.show()

plt.imsave(f"D:\desktop\SVD/Image_1({k}).jpg", compressed_image)    #输入存储压缩后图片路径与名称
