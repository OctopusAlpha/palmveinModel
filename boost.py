import cv2
import numpy as np
from PIL import Image
import os

def enhance_image(image):
    """
    对图像进行增强处理，包括中值滤波降噪和CLAHE对比度增强
    Args:
        image: 输入图像，可以是PIL.Image对象、numpy数组或图像路径
    Returns:
        PIL.Image: 增强后的图像
    """
    # 如果输入是路径，读取图像
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f'无法读取图像: {image}')
    # 如果输入是PIL.Image，转换为numpy数组
    elif isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 应用中值滤波进行降噪
    # 使用5x5的核大小，可以根据需要调整
    denoised = cv2.medianBlur(gray, 5)
    
    # 创建CLAHE对象
    # clipLimit控制对比度的限制阈值，tileGridSize定义每个网格的大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # 应用CLAHE进行对比度增强
    enhanced = clahe.apply(denoised)
    
    # 转换回PIL图像
    enhanced_pil = Image.fromarray(enhanced)
    
    return enhanced_pil

def test_enhancement(image_path):
    """
    测试图像增强效果
    Args:
        image_path (str): 图像路径
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 读取原始图像
    original = cv2.imread(image_path)
    if original is None:
        print(f'无法读取图像: {image_path}')
        return
    
    # 创建ROI提取器并提取ROI
    from preprocess import ROIExtractor
    roi_extractor = ROIExtractor()
    roi = roi_extractor.extract_roi(original)
    
    if roi is None:
        print('ROI提取失败')
        return
    
    # 进行图像增强
    enhanced_pil = enhance_image(roi)
    enhanced = np.array(enhanced_pil)
    
    # 显示原始图像、ROI区域和增强后的图像
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(15, 7))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title('ROI区域')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(enhanced, cmap='gray')
    plt.title('增强后的ROI')
    plt.axis('off')
    
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, f'roi_result_{os.path.basename(image_path)}')
        plt.savefig(save_path)
        print(f'结果已保存到: {save_path}')
    plt.show()

if __name__ == '__main__':
    # 测试示例
    image_path = 'dataset/train/001/00001.tiff'  # 替换为实际的图像路径
    save_dir = 'results/roi_boost'  # 替换为实际的保存目录路径
    test_enhancement(image_path)