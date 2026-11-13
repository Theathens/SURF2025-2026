import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys


# 加载ResNet-50模型用于识别
def load_resnet50_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model


# 图像预处理
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


# 图像识别
def classify_image(model, input_batch, categories, n):
    with torch.no_grad():
        output = model(input_batch)
    probabilities = F.softmax(output[0], dim=0)
    topn_prob, topn_catid = torch.topk(probabilities, n)
    return topn_prob, topn_catid


# SVD压缩
def svd_compression(image_path, k):
    image = Image.open(image_path)
    image = np.array(image)
    compressed_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[2]):
        U, Sigma, VT = np.linalg.svd(image[:, :, i])
        compressed_image[:, :, i] = U[:, :k] @ np.diag(Sigma[:k]) @ VT[:k, :]
    compressed_image = compressed_image.astype(np.uint8)
    return compressed_image


# 处理文件夹中的所有图像
def process_folder(input_folder, output_folder, k, model, categories, n):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path):
            try:
                with Image.open(input_path) as img:
                    # 图像识别
                    input_batch = preprocess_image(input_path)
                    topn_prob, topn_catid = classify_image(model, input_batch, categories, n)

                    # 保存识别结果
                    output_txt_path = os.path.join(output_folder,
                                                   f"recognition_result_{os.path.splitext(filename)[0]}.txt")
                    with open(output_txt_path, "w") as f:
                        f.write(f"识别结果 for {filename}:\n")
                        for i in range(n):
                            f.write(f"Top {i + 1}: {categories[topn_catid[i]]} (Possibility: {topn_prob[i].item():.4f})\n")
                    print(f"识别结果已保存到：{output_txt_path}")

                    # 图像压缩
                    compressed_image = svd_compression(input_path, k)
                    output_img_path = os.path.join(output_folder, f"compressed_{filename}")
                    plt.imsave(output_img_path, compressed_image)
                    print(f"压缩图像已保存到：{output_img_path}")
            except Exception as e:
                print(f"跳过文件 {filename}，因为它不是有效的图像文件或无法处理: {e}")


def main():
    # input_folder = input("请输入输入文件夹路径: ").strip('"')
    # output_folder = input("请输入输出文件夹路径: ").strip('"')
    input_folder = "C:/Users/27785/Desktop/recognize+compress/input"
    output_folder = "C:/Users/27785/Desktop/recognize+compress/output(5)(1)"
    k = int(input("请输入k值: "))
    n = int(input("请输入要显示的前n个类别的数量: "))

    if not os.path.isdir(input_folder):
        print("错误：输入路径不是一个有效的文件夹路径。")
        sys.exit(1)

    # 加载预训练的ResNet-50模型和类别文件
    model = load_resnet50_model()
    with open(r"C:\Users\27785\Desktop\recognize+compress\imagenet_classes.txt", "r") as f:  # 读取数据集
        categories = [s.strip() for s in f.readlines()]

    # 处理文件夹中的所有图像
    process_folder(input_folder, output_folder, k, model, categories, n)



if __name__ == "__main__":
    main()