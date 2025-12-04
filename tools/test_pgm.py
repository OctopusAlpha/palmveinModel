import cv2
from roi_extraction import ROIExtractor
import matplotlib.pyplot as plt

# 读取PGM图像
image_path = 'saved_image.pgm'
image = cv2.imread(image_path)

if image is None:
    print(f'无法读取图像: {image_path}')
    exit(1)

# 创建ROI提取器并提取ROI
roi_extractor = ROIExtractor()
palm_center, roi_resized, rect_coords = roi_extractor.extract_roi(image, visualize=True)

if roi_resized is not None:
    print('ROI提取成功')
    print(f'掌心中心点坐标: {palm_center}')
    print(f'ROI区域坐标: {rect_coords}')
else:
    print('ROI提取失败')