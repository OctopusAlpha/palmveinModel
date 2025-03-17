import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
plt.rcParams['font.sans-serif'] = ['SimHei']


class ROIExtractor:
    """
    ROI提取器，用于提取掌静脉ROI区域
    """
    def __init__(self, roi_size=(224, 224)):
        """
        初始化ROI提取器
        Args:
            roi_size (tuple): ROI区域的大小 (width, height)
        """
        self.roi_size = roi_size
    
    def extract_roi(self, image, visualize=True):
        # 1. 图像预处理：中值滤波去噪
        median_filtered = cv2.medianBlur(image, 5)
        
        # 转换为灰度图
        gray = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2GRAY)
        
        # 2. 固定阈值二值化
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        
        # 3. 距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # 归一化距离变换结果到0-1范围
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # 根据距离最大值的一定比例设置阈值
        alpha = 0.6  # 阈值系数
        max_dist = dist_transform.max()
        threshold_dist = alpha * max_dist
        
        # 阈值化得到掌心区域
        _, palm_center_region = cv2.threshold(dist_transform, threshold_dist, 1.0, cv2.THRESH_BINARY)
        palm_center_region = palm_center_region.astype(np.uint8) * 255
        
        # 找到掌心中心点（距离最大的点）
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
        
        # 定义边缘安全区域（图像边缘的安全距离）
        edge_margin = max(self.roi_size[0] // 2, self.roi_size[1] // 2)  # 使用ROI尺寸的一半作为安全边距
        
        # 检查最大值点是否在安全区域内
        img_height, img_width = image.shape[:2]
        is_safe = (max_loc[0] >= edge_margin and 
                  max_loc[0] < img_width - edge_margin and 
                  max_loc[1] >= edge_margin and 
                  max_loc[1] < img_height - edge_margin)
        
        if is_safe:
            palm_center = max_loc  # 如果在安全区域内，使用最大值点作为掌心中心点
        else:
            # 如果最大值点在边缘区域，寻找安全区域内的最大值点
            # 创建安全区域掩码
            safe_mask = np.zeros_like(dist_transform)
            safe_mask[edge_margin:img_height-edge_margin, edge_margin:img_width-edge_margin] = 1
            
            # 在安全区域内寻找最大值点
            masked_dist = dist_transform * safe_mask
            _, _, _, safe_max_loc = cv2.minMaxLoc(masked_dist)
            palm_center = safe_max_loc
            print(f"原始掌心中心点{max_loc}位于边缘区域，已调整为安全区域内的点{palm_center}")
        
        # 4. 找到掌心区域的轮廓
        contours, _ = cv2.findContours(palm_center_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 找到最大的轮廓（掌心区域）
        if len(contours) > 0:
            palm_contour = max(contours, key=cv2.contourArea)
            
            # 计算掌心区域的最小外接矩形
            x, y, w, h = cv2.boundingRect(palm_contour)
            
            # 计算掌心区域的面积
            palm_area = cv2.contourArea(palm_contour)
            
            # 5. 计算以掌心中心点为中心的矩形
            # 初始化矩形大小为ROI尺寸的一半
            rect_width = min(self.roi_size[0] // 2, w // 2)
            rect_height = min(self.roi_size[1] // 2, h // 2)
            
            # 确保矩形完全在掌心区域内
            # 创建一个掌心区域的掩码
            mask = np.zeros_like(palm_center_region)
            cv2.drawContours(mask, [palm_contour], 0, 255, -1)
            
            # 计算矩形的左上角和右下角坐标
            rect_x1 = palm_center[0] - rect_width
            rect_y1 = palm_center[1] - rect_height
            rect_x2 = palm_center[0] + rect_width
            rect_y2 = palm_center[1] + rect_height
            
            # 确保矩形不超出图像边界
            rect_x1 = max(0, rect_x1)
            rect_y1 = max(0, rect_y1)
            rect_x2 = min(image.shape[1] - 1, rect_x2)
            rect_y2 = min(image.shape[0] - 1, rect_y2)
            
            # 提取ROI区域
            roi = image[rect_y1:rect_y2, rect_x1:rect_x2].copy()
            
            # 调整ROI大小为指定尺寸
            roi_resized = cv2.resize(roi, self.roi_size)
            
            # 在原图上绘制矩形框
            image_with_rect = image.copy()
            cv2.rectangle(image_with_rect, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)
            
            # 如果需要可视化，显示处理过程
            if visualize:
                plt.figure(figsize=(20, 5))
                
                # 显示原始图像
                plt.subplot(161)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title('原始图像')
                plt.axis('off')
                
                # 显示中值滤波结果
                plt.subplot(162)
                plt.imshow(gray, cmap='gray')
                plt.title('中值滤波结果')
                plt.axis('off')
                
                # 显示二值化结果
                plt.subplot(163)
                plt.imshow(binary, cmap='gray')
                plt.title('二值化结果')
                plt.axis('off')
                
                # 显示距离变换结果
                plt.subplot(164)
                plt.imshow(dist_transform, cmap='jet')
                plt.colorbar()
                plt.title('距离变换')
                plt.axis('off')
                
                # 显示掌心区域
                plt.subplot(165)
                plt.imshow(palm_center_region, cmap='gray')
                plt.plot(palm_center[0], palm_center[1], 'r+', markersize=10)
                plt.title('掌心区域')
                plt.axis('off')
                
                # 显示带矩形框的结果
                plt.subplot(166)
                plt.imshow(cv2.cvtColor(image_with_rect, cv2.COLOR_BGR2RGB))
                plt.title('ROI矩形框')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
                
                # 显示提取的ROI区域
                plt.figure(figsize=(5, 5))
                plt.imshow(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB))
                plt.title('提取的ROI区域')
                plt.axis('off')
                plt.show()
            
            return palm_center, roi_resized, (rect_x1, rect_y1, rect_x2, rect_y2)
        
        # 如果没有找到掌心区域，返回None
        return palm_center, None, None

def visualize_roi(image, roi, save_path=None):
    """
    并排显示原始图像和ROI提取结果
    Args:
        image (numpy.ndarray): 原始图像，BGR格式
        roi (numpy.ndarray): 提取的ROI区域，BGR格式
        save_path (str, optional): 可视化结果保存路径
    """
    # 创建图像窗口
    plt.figure(figsize=(12, 5))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示ROI区域
    plt.subplot(1, 2, 2)
    if roi is not None:
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        plt.title('ROI区域')
    else:
        plt.title('ROI提取失败')
    plt.axis('off')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存结果
    if save_path:
        plt.savefig(save_path)
    
    # 显示结果
    plt.show()

def test_roi_extraction():
    """测试ROI提取功能"""
    # 创建ROI提取器
    roi_extractor = ROIExtractor()
    os.makedirs("./results", exist_ok=True)
    # 测试图片路径
    test_images = [
        "dataset/train/199/07988.tiff",
        "dataset/train/001/06001.tiff"
    ]
    
    # 测试每张图片
    for image_path in test_images:
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片：{image_path}")
            continue
        
        # 提取ROI并可视化结果
        palm_center, roi, rect_coords = roi_extractor.extract_roi(image)
        if palm_center is not None:
            print(f"掌心中心点坐标：{palm_center}")
            
            # 如果成功提取ROI，保存结果
            if roi is not None:
                # 获取文件名（不含路径和扩展名）
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                # 保存ROI图像
                roi_save_path = f"./results/{file_name}_roi.png"
                cv2.imwrite(roi_save_path, roi)
                print(f"ROI已保存至：{roi_save_path}")
                
                # 可视化并保存对比图
                visualize_roi(image, roi, f"./results/{file_name}_comparison.png")
            else:
                print("ROI提取失败")
        else:
            print(f"ROI提取失败：{image_path}")

if __name__ == "__main__":
    test_roi_extraction()