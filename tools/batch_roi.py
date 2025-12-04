import os
import cv2
import numpy as np
from tqdm import tqdm
from src.core.roi import ROIExtractor
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_image(image_path, output_path):
    """
    处理单个图像并提取ROI
    Args:
        image_path (str): 输入图像路径
        output_path (str): 输出图像路径
    Returns:
        tuple: (是否成功, 图像路径, 错误信息)
    """
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return False, image_path, "无法读取图像"
        
        # 创建ROI提取器并提取ROI
        roi_extractor = ROIExtractor()
        palm_center, roi_resized, rect_coords = roi_extractor.extract_roi(image, visualize=False)
        
        if roi_resized is None:
            return False, image_path, "ROI提取失败"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存ROI图像
        cv2.imwrite(output_path, roi_resized)
        return True, image_path, "成功"
        
    except Exception as e:
        return False, image_path, str(e)

def batch_process_images(input_dir, output_dir, max_workers=None):
    """
    批量处理图像并提取ROI
    Args:
        input_dir (str): 输入目录
        output_dir (str): 输出目录
        max_workers (int, optional): 最大进程数
    """
    # 收集所有图像文件
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_path = os.path.join(root, file)
                rel_path = os.path.relpath(image_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                image_files.append((image_path, output_path))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录处理结果
    success_count = 0
    failed_files = []
    
    # 使用进度条显示处理进度
    with tqdm(total=len(image_files), desc="处理进度") as pbar:
        # 使用多进程处理图像
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_path = {executor.submit(process_image, img_path, out_path): img_path 
                             for img_path, out_path in image_files}
            
            # 处理结果
            for future in as_completed(future_to_path):
                success, image_path, message = future.result()
                if success:
                    success_count += 1
                else:
                    failed_files.append((image_path, message))
                pbar.update(1)
    
    # 打印处理结果
    print(f"\n处理完成！")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {len(failed_files)} 个文件")
    
    # 如果有失败的文件，保存到日志文件
    if failed_files:
        log_path = os.path.join(output_dir, 'failed_files.log')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("处理失败的文件列表:\n")
            for path, error in failed_files:
                f.write(f"{path}: {error}\n")
        print(f"失败文件列表已保存到: {log_path}")

if __name__ == '__main__':
    # 设置输入和输出目录
    input_dir = 'dataset'
    output_dir = 'dataset_roi'
    
    # 开始批量处理
    batch_process_images(input_dir, output_dir)