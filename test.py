import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.model.network import PalmVeinNet
from src import config
from src.core.enhance import enhance_image
from src.core.roi import ROIExtractor
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """
    加载训练好的模型
    Args:
        model_path (str): 模型文件路径
    Returns:
        model: 加载好的模型
    """
    # Ensure we match the model definition in src.model.network.PalmVeinNet
    model = PalmVeinNet(embedding_size=config.feature_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def preprocess_image(image_path, visualize=False):
    """
    预处理图像，与训练时保持一致，但移除随机变换以保持测试一致性
    首先提取ROI区域，然后应用CLAHE增强
    Args:
        image_path (str): 图像文件路径
        visualize (bool): 是否可视化ROI提取过程
    Returns:
        torch.Tensor: 预处理后的图像张量
    """
    # 使用确定性预处理，移除随机变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计值
                            std=[0.229, 0.224, 0.225])
    ])
    
    # 读取原始图像
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f'无法读取图像: {image_path}')
    
    # 创建ROI提取器并提取ROI
    roi_extractor = ROIExtractor()
    # Note: extract_roi signature might return different things depending on implementation
    # Let's check src/core/roi.py if possible, but assuming it returns (center, roi, rect)
    try:
        palm_center, roi_resized, rect_coords = roi_extractor.extract_roi(original, visualize=visualize)
    except:
        # Fallback if signature is different or fails
        roi_resized = None
    
    if roi_resized is None:
        print(f'ROI提取失败，使用原始图像: {image_path}')
        # 如果ROI提取失败，使用原始图像
        if len(original.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original
        pil_image = Image.fromarray(gray)
    else:
        # 转换ROI为灰度图
        if len(roi_resized.shape) == 3:
            gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi_resized
        pil_image = Image.fromarray(gray_roi)
    
    # 应用CLAHE增强，与训练时一致
    enhanced_image = enhance_image(pil_image)
    
    # 将灰度图像转换为三通道
    image = Image.merge('RGB', (enhanced_image, enhanced_image, enhanced_image))
    
    # 应用变换
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image

def extract_features(model, image):
    """
    提取图像特征向量
    Args:
        model: 模型
        image (torch.Tensor): 预处理后的图像张量
    Returns:
        torch.Tensor: 特征向量
    """
    device = next(model.parameters()).device
    image = image.to(device)
    
    # 确保模型处于评估模式
    model.eval()
    
    # 处理BatchNorm层在单样本输入时的问题
    with torch.no_grad():
        if image.size(0) == 1:
            # 对于单个样本，复制一次形成批次大小为2的输入
            batch_image = torch.cat([image, image], dim=0)
            features = model(batch_image)
            # 只取第一个样本的特征
            features = features[0:1]
        else:
            features = model(image)
    return features

def cosine_similarity(features1, features2):
    """
    计算两个特征向量的余弦相似度
    Args:
        features1 (torch.Tensor): 第一个特征向量
        features2 (torch.Tensor): 第二个特征向量
    Returns:
        float: 余弦相似度
    """
    return F.cosine_similarity(features1, features2).item()

def SingleCompare():
    # 设置模型路径
    best_model = f"{config.save_model_dir}/best_palm_vein_model.pth" # Updated name
    if not os.path.exists(best_model):
        print(f"Model not found at {best_model}")
        return

    print(f'使用模型: {best_model}')
    
    # 加载模型
    model = load_model(best_model)
    
    # 设置测试图像路径 - update these paths to valid ones on your system if needed
    # For now, just examples
    valid_dir = config.valid_dir
    if not os.path.exists(valid_dir):
        print(f"Validation directory not found: {valid_dir}")
        return

    # Try to find some images automatically
    classes = [d for d in os.listdir(valid_dir) if os.path.isdir(os.path.join(valid_dir, d))]
    if not classes:
        print("No classes found in validation dir.")
        return
        
    c1 = classes[0]
    imgs1 = os.listdir(os.path.join(valid_dir, c1))
    if len(imgs1) < 2:
        print("Not enough images for testing.")
        return
        
    image_path1 = os.path.join(valid_dir, c1, imgs1[0])
    image_path2 = os.path.join(valid_dir, c1, imgs1[1]) # Same person
    
    print(f'Testing with:\n1: {image_path1}\n2: {image_path2}')
    
    # 预处理图像（包括ROI提取和增强）
    print('处理第一张图像...')
    image1 = preprocess_image(image_path1, visualize=False)
    print('处理第二张图像...')
    image2 = preprocess_image(image_path2, visualize=False)
    
    # 提取特征
    features1 = extract_features(model, image1)
    features2 = extract_features(model, image2)
    
    # 计算相似度
    similarity = cosine_similarity(features1, features2)
    print(f'相似度: {similarity:.4f}')
    
    # 设置相似度阈值
    threshold = 0.8
    if similarity > threshold:
        print('匹配: 同一个人的掌静脉')
    else:
        print('不匹配: 不同人的掌静脉')

if __name__ == '__main__':
    SingleCompare()
