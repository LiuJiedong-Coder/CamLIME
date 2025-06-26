import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage import io
from lime.wrappers.scikit_image import SegmentationAlgorithm

# 加载示例图像
image = io.imread('test_img/ILSVRC2012_val_00011333.JPEG')  # 替换为实际图像路径
image = image / 255.0  # 将图像归一化到 [0, 1]

# 创建分割器
segmenter = SegmentationAlgorithm('slic', n_segments=40, compactness=1, sigma=1)

# 对图像进行分割
segments = segmenter(image)

# 可视化分割图像并添加编号
fig, ax = plt.subplots(figsize=(5, 5))
segmented_image = mark_boundaries(image, segments)
ax.imshow(segmented_image)

# 获取分割块编号的中心位置
unique_segments = np.unique(segments)
for segment_id in unique_segments:
    # 获取当前分割块的像素坐标
    y_coords, x_coords = np.where(segments == segment_id)
    # 计算中心点
    center_x, center_y = x_coords.mean(), y_coords.mean()
    # 在中心点标注分割块编号
    ax.text(center_x, center_y, str(segment_id), color='red', fontsize=8, ha='center', va='center')

# 设置标题和隐藏坐标轴
ax.set_title("Segmented Image with Feature Numbers", fontsize=14)
ax.axis("off")

plt.tight_layout()
plt.show()
