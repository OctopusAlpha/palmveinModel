import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


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
    
    def repair_binary_image(self, binary_image, original_gray=None, mode="morphology+contour", kernel_size=5, min_area=100, visualize=False):
        """
        对二值化后的图像进行修复
        Args:
            binary_image: 二值化后的图像
            original_gray: 原图灰度版本（可选）
            mode: 修复模式组合，例如 "morphology+contour"
            kernel_size: 形态学核大小
            min_area: 最小连通区域面积阈值
            visualize: 是否可视化修复过程
        Returns:
            修复后的二值图像
        """
        # 保存每个步骤的结果用于可视化
        steps = [("原始二值图像", binary_image.copy())]
        # 创建结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 1. 形态学操作
        if "morphology" in mode:
            # 闭运算填充孔洞
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            steps.append(("闭运算结果", binary_image.copy()))
            # 开运算去噪
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            steps.append(("开运算结果", binary_image.copy()))
        
        # 2. 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        
        # 创建掩码，只保留大于最小面积的连通域
        mask = np.zeros_like(binary_image)
        for i in range(1, num_labels):  # 跳过背景（标签0）
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == i] = 255
        
        binary_image = mask
        steps.append(("连通域分析结果", binary_image.copy()))
        
        # 3. 轮廓修复
        if "contour" in mode:
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # 计算轮廓的凸包
                hull = cv2.convexHull(contour)
                # 绘制凸包
                cv2.drawContours(binary_image, [hull], 0, 255, -1)
            steps.append(("轮廓修复结果", binary_image.copy()))
        
        # 4. 与原始灰度图像融合
        if original_gray is not None:
            # 使用原始灰度图像的信息来调整二值图像
            # 在二值图像为255的区域，保留原始灰度值
            binary_image = np.where(binary_image == 255, original_gray, 0)
            # 重新二值化
            _, binary_image = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            steps.append(("与原图融合结果", binary_image.copy()))
        
        # 可视化修复过程
        if visualize:
            n_steps = len(steps)
            n_cols = min(3, n_steps)  # 每行最多3个子图
            n_rows = (n_steps + n_cols - 1) // n_cols  # 计算需要的行数
            
            plt.figure(figsize=(15, 5 * n_rows))
            for idx, (title, img) in enumerate(steps, 1):
                plt.subplot(n_rows, n_cols, idx)
                plt.imshow(img, cmap='gray')
                plt.title(title)
                plt.axis('off')
            
            plt.suptitle('图像修复过程可视化', fontsize=16)
            plt.tight_layout()
            plt.show()
        
        return binary_image
    
    def extract_roi(self, image, visualize=False):
        # 1. 图像预处理：中值滤波去噪
        median_filtered = cv2.medianBlur(image, 5)
        
        # 转换为灰度图
        gray = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2GRAY)
        
        # 2. 自适应阈值二值化
        block_size = 33  # 邻域大小，必须是奇数
        C = 5  # 常数，从计算出的平均值中减去的值
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
        
        # 修复二值化图像
        binary = self.repair_binary_image(binary, gray, mode="morphology+contour", kernel_size=5, min_area=100)
        
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
        palm_center = max_loc
        # 定义边缘安全区域（图像边缘的安全距离）
        # edge_margin = max(self.roi_size[0] // 2, self.roi_size[1] // 2)  # 使用ROI尺寸的一半作为安全边距
        
        # # 检查最大值点是否在安全区域内
        # img_height, img_width = image.shape[:2]
        # is_safe = (max_loc[0] >= edge_margin and 
        #           max_loc[0] < img_width - edge_margin and 
        #           max_loc[1] >= edge_margin and 
        #           max_loc[1] < img_height - edge_margin)
        
        # if is_safe:
        #     palm_center = max_loc  # 如果在安全区域内，使用最大值点作为掌心中心点
        # else:
        #     # 如果最大值点在边缘区域，寻找安全区域内的最大值点
        #     # 创建安全区域掩码
        #     safe_mask = np.zeros_like(dist_transform)
        #     safe_mask[edge_margin:img_height-edge_margin, edge_margin:img_width-edge_margin] = 1
            
        #     # 在安全区域内寻找最大值点
        #     masked_dist = dist_transform * safe_mask
        #     _, _, _, safe_max_loc = cv2.minMaxLoc(masked_dist)
        #     palm_center = safe_max_loc
        #     print(f"原始掌心中心点{max_loc}位于边缘区域，已调整为安全区域内的点{palm_center}")
        
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
                # 1. 显示原始图像和预处理结果
                plt.figure(figsize=(15, 10))
                plt.subplot(231)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title('1. 原始图像')
                plt.axis('off')
                
                plt.subplot(232)
                plt.imshow(gray, cmap='gray')
                plt.title('2. 中值滤波结果')
                plt.axis('off')
                
                plt.subplot(233)
                plt.imshow(binary, cmap='gray')
                plt.title('3. 自适应二值化结果')
                plt.axis('off')
                
                # 2. 显示距离变换和掌心检测结果
                plt.subplot(234)
                plt.imshow(dist_transform, cmap='jet')
                plt.colorbar()
                plt.title('4. 距离变换')
                plt.axis('off')
                
                plt.subplot(235)
                plt.imshow(palm_center_region, cmap='gray')
                plt.plot(palm_center[0], palm_center[1], 'r+', markersize=10, label='掌心中心点')
                plt.title('5. 掌心区域检测')
                plt.legend()
                plt.axis('off')
                
                plt.subplot(236)
                plt.imshow(cv2.cvtColor(image_with_rect, cv2.COLOR_BGR2RGB))
                plt.plot(palm_center[0], palm_center[1], 'r+', markersize=10, label='掌心中心点')
                plt.title('6. ROI区域定位')
                plt.legend()
                plt.axis('off')
                
                plt.suptitle('ROI提取过程可视化', fontsize=16)
                plt.tight_layout()
                plt.show()
                
                # 3. 显示最终提取的ROI结果
                plt.figure(figsize=(6, 6))
                plt.imshow(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB))
                plt.title('最终提取的ROI区域')
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