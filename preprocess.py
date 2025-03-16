import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
class HandKeypointDetector:
    """
    手掌关键点检测器，用于定位手掌上的关键点
    """
    def __init__(self):
        # 初始化关键点检测器
        self.detector = cv2.SimpleBlobDetector_create(self._get_detector_params())
        
    def _get_detector_params(self):
        """
        设置关键点检测器的参数
        Returns:
            cv2.SimpleBlobDetector_Params: 检测器参数
        """
        params = cv2.SimpleBlobDetector_Params()
        
        # 根据面积过滤
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 5000
        
        # 根据圆度过滤
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # 根据凸度过滤
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # 根据惯性比过滤
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        
        return params
    
    def detect_keypoints(self, image):
        """
        检测手掌关键点
        Args:
            image (numpy.ndarray): 输入图像，BGR格式
        Returns:
            tuple: (关键点A, 关键点B, 关键点C)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 图像预处理
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 检测手掌轮廓
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        
        # 找到最大轮廓（假设是手掌）
        palm_contour = max(contours, key=cv2.contourArea)
        
        # 计算凸包和凸缺陷
        hull = cv2.convexHull(palm_contour, returnPoints=False)
        defects = cv2.convexityDefects(palm_contour, hull)
        
        if defects is None:
            return None, None, None
        
        # 提取指间隙点
        finger_valleys = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(palm_contour[s][0])
            end = tuple(palm_contour[e][0])
            far = tuple(palm_contour[f][0])
            
            # 过滤掉不是指间隙的凸缺陷
            if d / 256.0 > 10:  # 深度阈值
                finger_valleys.append(far)
        
        # 如果检测到的指间隙点少于3个，返回None
        if len(finger_valleys) < 3:
            return None, None, None
        
        # 按照y坐标排序（从上到下）
        finger_valleys.sort(key=lambda p: p[1])
        
        # 取前三个点作为A, B, C点（食指与中指间隙、中指与无名指间隙、无名指与小指间隙）
        # 按照x坐标排序（从左到右）
        top_valleys = sorted(finger_valleys[:3], key=lambda p: p[0])
        
        # 返回三个关键点
        point_a = top_valleys[0] if len(top_valleys) > 0 else None
        point_b = top_valleys[1] if len(top_valleys) > 1 else None
        point_c = top_valleys[2] if len(top_valleys) > 2 else None
        
        return point_a, point_b, point_c

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
        self.keypoint_detector = HandKeypointDetector()
    
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
        palm_center = max_loc  # 掌心中心点坐标
        
        # 使用Canny边缘检测
        edge = cv2.Canny(binary, 50, 150)
        
        # 形态学操作以连接边缘
        kernel = np.ones((3,3), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
        
        # 连接手指尖和手掌根部的断开区域
        # 找到最上面和最下面的点
        y_coords = np.where(edge > 0)[0]
        if len(y_coords) > 0:
            top_y = y_coords.min()
            bottom_y = y_coords.max()
            
            # 对于每一行，找到最左边和最右边的点
            for y in [top_y, bottom_y]:  # 只处理最上面和最下面的行
                x_coords = np.where(edge[y, :] > 0)[0]
                if len(x_coords) >= 2:
                    left_x = x_coords.min()
                    right_x = x_coords.max()
                    # 在两点之间画一条直线
                    cv2.line(edge, (left_x, y), (right_x, y), 255, 2)
        
        # 3. 寻找轮廓
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("未检测到轮廓")
            if visualize:
                visualize_roi(image, None)
            return None
        
        # 找到最大轮廓（假设是手掌）
        palm_contour = max(contours, key=cv2.contourArea)
        
        # 拟合外切圆
        (x, y), radius = cv2.minEnclosingCircle(palm_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
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
            
            # 显示带有轮廓和外切圆的图像
            plt.subplot(166)
            vis_image = image.copy()
            cv2.drawContours(vis_image, [palm_contour], -1, (255, 0, 0), 2)  # 蓝色轮廓
            cv2.circle(vis_image, center, radius, (0, 255, 0), 2)  # 绿色外切圆
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title('轮廓和外切圆')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # 3. 寻找轮廓
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("未检测到轮廓")
            if visualize:
                visualize_roi(image, None)
            return None
        
        # 找到最大轮廓（假设是手掌）
        palm_contour = max(contours, key=cv2.contourArea)
        
        # 拟合外切圆
        (x, y), radius = cv2.minEnclosingCircle(palm_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 计算轮廓点到圆心的距离
        distances = [np.sqrt((p[0][0]-x)**2 + (p[0][1]-y)**2) for p in palm_contour]
        distances = np.array(distances)
        
        # 找到与外切圆相交的点（角度点）
        threshold = 0.95  # 距离阈值（相对于半径）
        angle_points = []
        for i, p in enumerate(palm_contour):
            if abs(distances[i] - radius) / radius < threshold:
                angle_points.append(p[0])
        
        # 如果找到的角度点少于2个，返回None
        if len(angle_points) < 2:
            if visualize:
                visualize_roi(image, None)
            return None
        
        # 4. 计算角度点的中垂线
        # 选择距离最远的两个角度点
        angle_points = np.array(angle_points)
        dist_matrix = np.sqrt(np.sum((angle_points[:, None] - angle_points[None, :])**2, axis=2))
        i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
        p1, p2 = angle_points[i], angle_points[j]
        
        # 计算中点
        mid_point = (int((p1[0] + p2[0])//2), int((p1[1] + p2[1])//2))
        
        # 计算中垂线斜率
        if p2[0] - p1[0] != 0:
            slope = -(p2[0] - p1[0]) / (p2[1] - p1[1]) if p2[1] - p1[1] != 0 else float('inf')
            angle = np.degrees(np.arctan(slope))
        else:
            angle = 90
            
        # 可视化角度点和中垂线
        if visualize:
            vis_image = image.copy()
            # 绘制角度点
            cv2.circle(vis_image, tuple(p1), 5, (0, 0, 255), -1)  # 红色
            cv2.circle(vis_image, tuple(p2), 5, (0, 255, 0), -1)  # 绿色
            
            # 计算中垂线的端点
            line_length = 100  # 中垂线长度
            if slope != float('inf'):
                dx = line_length / np.sqrt(1 + slope**2)
                dy = slope * dx
                pt1 = (int(mid_point[0] - dx), int(mid_point[1] - dy))
                pt2 = (int(mid_point[0] + dx), int(mid_point[1] + dy))
            else:
                pt1 = (mid_point[0], mid_point[1] - line_length)
                pt2 = (mid_point[0], mid_point[1] + line_length)
            
            # 绘制中垂线
            cv2.line(vis_image, pt1, pt2, (255, 0, 0), 2)  # 蓝色
            
            # 显示结果
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title('角度点和中垂线')
            plt.axis('off')
            plt.show()
        
        # 旋转校正
        rotation_matrix = cv2.getRotationMatrix2D(mid_point, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        # 提取ROI区域
        roi_size = min(self.roi_size)
        x1 = max(0, mid_point[0] - roi_size//2)
        y1 = max(0, mid_point[1] - roi_size//2)
        x2 = min(image.shape[1], x1 + roi_size)
        y2 = min(image.shape[0], y1 + roi_size)
        
        roi = rotated[y1:y2, x1:x2]
        
        # # 显示可视化结果
        # if visualize:
        #     visualize_roi(image, roi)
        
        return roi
    
    def visualize_keypoints(self, image, save_path=None):
        """
        可视化关键点和ROI区域
        Args:
            image (numpy.ndarray): 输入图像
            save_path (str, optional): 保存路径
        """
        # 复制图像以避免修改原图
        vis_image = image.copy()
        
        # 在需要时创建HandKeypointDetector实例
        keypoint_detector = HandKeypointDetector()
        
        # 检测关键点
        point_a, point_b, point_c = keypoint_detector.detect_keypoints(image)
        
        # 如果关键点检测成功，绘制关键点和连线
        if point_a is not None and point_b is not None and point_c is not None:
            # 绘制关键点
            cv2.circle(vis_image, point_a, 5, (0, 0, 255), -1)  # 红色
            cv2.circle(vis_image, point_b, 5, (0, 255, 0), -1)  # 绿色
            cv2.circle(vis_image, point_c, 5, (255, 0, 0), -1)  # 蓝色
            
            # 绘制连线
            cv2.line(vis_image, point_a, point_c, (255, 255, 0), 2)  # 黄色
            
            # 提取ROI
            roi = self.extract_roi(image)
            
            # 显示图像
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title('Detected Keypoints')
            
            if roi is not None:
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                plt.title('Extracted ROI')
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
        else:
            print("关键点检测失败")

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

def preprocess_image(image_path, roi_extractor=None):
    """
    预处理图像，包括关键点定位和ROI提取
    Args:
        image_path (str): 图像路径
        roi_extractor (ROIExtractor, optional): ROI提取器
    Returns:
        PIL.Image: 预处理后的图像，如果处理失败则返回原图
    """
    # 如果没有提供ROI提取器，创建一个新的
    if roi_extractor is None:
        roi_extractor = ROIExtractor()

def test_roi_extraction():
    """测试ROI提取功能"""
    # 创建ROI提取器
    roi_extractor = ROIExtractor()
    os.makedirs("./results", exist_ok=True)
    # 测试图片路径
    test_images = [
        "dataset/train/001/00008.tiff",
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
        roi = roi_extractor.extract_roi(image)
        if roi is not None:
            visualize_roi(image, roi, save_path=f"results/{image_path.split('/')[-1].replace('.tiff', '_roi.png')}")
        else:
            print(f"ROI提取失败：{image_path}")

if __name__ == "__main__":
    test_roi_extraction()