import matplotlib.pyplot as plt
import numpy as np

# 1. 生成数据（Epoch 0-50，k值固定为22.0）
x = np.arange(1, 51)  # x轴：Epoch从0到50
y = np.full_like(x, 22.0)  # y轴：所有点的k值均为22.0

# 2. 创建画布并绘制图表
plt.figure(figsize=(10, 6))  # 调整画布大小以匹配原图比例

# 绘制折线图（蓝色圆点，无数据点标注）
plt.plot(
    x, y,
    marker='o',        # 圆点标记
    color='blue',      # 蓝色
    linestyle='-',     # 实线连接
    markersize=4       # 标记大小
)

# 3. 设置标题和轴标签
plt.title('Average k Value per Epoch', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Average k Value', fontsize=10)

# 4. 调整坐标轴刻度
plt.xticks(np.arange(0, 51, 10))  # x轴每10个Epoch标记一次
plt.yticks(np.arange(21.0, 23.1, 0.5))  # y轴范围21.0-23.0，步长0.5

# 5. 添加网格线（与原图风格一致：灰色虚线）
plt.grid(
    True,
    linestyle='--',  # 虚线
    color='gray',    # 灰色
    alpha=0.7        # 透明度
)

# 6. 调整边框样式（隐藏上、右、左边框）
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# 7. 优化布局并保存图片
plt.tight_layout()  # 自动调整布局，避免元素重叠
plt.savefig(
    'average_k_value_fixed.png',
    dpi=300,          # 高分辨率
    bbox_inches='tight'  # 确保图片无截断
)
plt.show()