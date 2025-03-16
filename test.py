import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import PalmVeinNet

def load_model(model_path):
    """
    加载训练好的模型
    Args:
        model_path (str): 模型文件路径
    Returns:
        model: 加载好的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PalmVeinNet(feature_dim=512).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image_path):
    """
    预处理图像
    Args:
        image_path (str): 图像文件路径
    Returns:
        torch.Tensor: 预处理后的图像张量
    """
    # 使用与训练时相同的预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    # 直接加载图像
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
            
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
    
    with torch.no_grad():
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

def main():
    # 设置模型路径（使用最新的模型文件）
    checkpoint_dir = 'checkpoints'
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not model_files:
        print('未找到模型文件')
        return
    
    # 获取最新的模型文件
    latest_model = max([os.path.join(checkpoint_dir, f) for f in model_files], key=os.path.getmtime)
    print(f'使用模型: {latest_model}')
    
    # 加载模型
    model = load_model(latest_model)
    
    # 设置测试图像路径
    image_path1 = 'dataset/valid/001/00004.tiff'
    # image_path2 = 'dataset/valid/002/00020.tiff'
    image_path2 = 'dataset/valid/001/00005.tiff'
    
    # 预处理图像
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)
    
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
    main()