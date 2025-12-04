import cv2
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 

# 读取图像
img = cv2.imread('dataset_roi\\train\\001\\00001.tiff', 0)  # 0表示以灰度模式读取

# 计算直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 显示直方图
plt.figure()
plt.title('灰度直方图')
plt.xlabel('灰度值')
plt.ylabel('像素数')
plt.plot(hist)
plt.xlim([0, 256])
plt.show()