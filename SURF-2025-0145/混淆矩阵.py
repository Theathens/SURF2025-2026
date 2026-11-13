import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 混淆矩阵数据（与之前一致）
confusion_matrix_data = np.array([
    [129, 1, 0, 0, 0, 0, 0, 0],  # density1Benign
    [1, 317, 0, 6, 0, 0, 0, 0],  # density1Malignant
    [0, 0, 43, 0, 0, 0, 0, 0],  # density2Benign
    [0, 0, 0, 346, 0, 0, 0, 0],  # density2Malignant
    [0, 0, 0, 0, 140, 0, 0, 0],  # density3Benign
    [0, 0, 0, 0, 0, 86, 0, 0],  # density3Malignant
    [0, 0, 0, 0, 0, 0, 65, 0],  # density4Benign
    [0, 0, 0, 0, 0, 0, 0, 11]  # density4Malignant
])

# 2. 完整标签
# full_labels = [
#     "density1Benign",
#     "density1Malignant",
#     "density2Benign",
#     "density2Malignant",
#     "density3Benign",
#     "density3Malignant",
#     "density4Benign",
#     "density4Malignant"
# ]

full_labels = [
    "d1B",
    "d1M",
    "d2B",
    "d2M",
    "d3B",
    "d3M",
    "d4B",
    "d4M",
]


# 3. 绘制混淆矩阵（修正参数错误）
def plot_complete_confusion_matrix(cm, labels, output_path="confusion_matrix_complete.png"):
    plt.figure(figsize=(16, 14))  # 画布大小

    # 绘制热图（移除错误参数 xticklabels_rotation 和 yticklabels_rotation）
    ax = sns.heatmap(
        cm,
        annot=True,  # 显示数值
        fmt='d',  # 整数格式
        cmap='Blues',  # 颜色主题
        xticklabels=labels,  # x轴标签
        yticklabels=labels,  # y轴标签
        annot_kws={"size": 25},  # 数值字体大小
        cbar_kws={"shrink": 0.8}  # 颜色条大小
    )

    # 单独设置标签旋转角度（正确方式）
    plt.xticks(rotation=45, ha='right')  # x轴标签旋转45度，右对齐
    plt.yticks(rotation=0)  # y轴标签不旋转

    # 设置标题和轴标签
    plt.title('Confusion Matrix (Validation Set) - Complete Labels', fontsize=16, pad=20)
    plt.xlabel('Predicted Class', fontsize=14, labelpad=15)
    plt.ylabel('True Class', fontsize=14, labelpad=15)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    print(f"完整标签的混淆矩阵已保存至：{os.path.abspath(output_path)}")


# 4. 执行绘图
if __name__ == "__main__":
    plot_complete_confusion_matrix(
        confusion_matrix_data,
        full_labels,
        output_path="complete_confusion_matrix.png"
    )
