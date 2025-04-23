import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import PalmVeinNet
import config
from train import ResNet18Embedder
from CLAHE import enhance_image
from roi_extraction import ROIExtractor
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
    model = ResNet18Embedder(embedding_size=128).to(device)  # 确保模型架构与训练时一致
    model.load_state_dict(torch.load(model_path))
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
        transforms.Resize(256),
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
    palm_center, roi_resized, rect_coords = roi_extractor.extract_roi(original, visualize=visualize)
    
    if roi_resized is None:
        print(f'ROI提取失败，使用原始图像: {image_path}')
        # 如果ROI提取失败，使用原始图像
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray)
    else:
        # 转换ROI为灰度图
        gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
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

def SigleCompare():
    # 设置模型路径
    best_model = f"{config.save_model_dir}/best_model.pth"
    print(f'使用模型: {best_model}')
    
    # 加载模型
    model = load_model(best_model)
    
    # 设置测试图像路径
    image_path1 = 'dataset/valid/001/00004.tiff'
    # image_path2 = 'dataset/valid/002/00020.tiff'
    # image_path2 = 'dataset/valid/001/00004.tiff'
    image_path2 = 'dataset\\valid\\004\\00031.tiff'
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
def BatchCompare(num_pairs=100, threshold=0.8, visualize=False):
    """
    批量比较掌静脉图像，计算准确率、精确率、召回率等指标
    Args:
        num_pairs (int): 要比较的图像对数量
        threshold (float): 相似度阈值，大于该值被认为是匹配
        visualize (bool): 是否可视化ROI提取过程
    Returns:
        dict: 包含各项评估指标的字典
    """
    # 设置模型路径
    best_model = f"{config.save_model_dir}/best_model.pth"
    print(f'使用模型: {best_model}')
    
    # 加载模型
    model = load_model(best_model)
    
    # 获取验证集目录
    valid_dir = 'dataset/valid'
    
    # 获取所有类别（每个人一个类别/文件夹）
    person_ids = [d for d in os.listdir(valid_dir) if os.path.isdir(os.path.join(valid_dir, d))]
    print(f'找到 {len(person_ids)} 个不同的人')
    
    # 初始化评估指标计数器
    true_positives = 0  # 真阳性：同一个人，预测为同一个人
    false_positives = 0  # 假阳性：不同人，预测为同一个人
    true_negatives = 0  # 真阴性：不同人，预测为不同人
    false_negatives = 0  # 假阴性：同一个人，预测为不同人
    
    # 记录所有相似度
    same_person_similarities = []
    diff_person_similarities = []
    
    # 创建相同人和不同人的图像对
    same_person_pairs = 0
    diff_person_pairs = 0
    
    # 确保相同人和不同人的图像对数量大致相等
    target_pairs_each = num_pairs // 2
    
    import random
    import time
    start_time = time.time()
    
    print(f'开始批量比较，目标生成 {num_pairs} 对图像...')
    
    # 1. 生成相同人的图像对
    while same_person_pairs < target_pairs_each and len(person_ids) > 0:
        # 随机选择一个人
        person_id = random.choice(person_ids)
        person_dir = os.path.join(valid_dir, person_id)
        
        # 获取该人的所有图像
        images = [f for f in os.listdir(person_dir) if f.endswith('.tiff')]
        
        # 如果该人有至少两张图像，则可以形成一对
        if len(images) >= 2:
            # 随机选择两张不同的图像
            img1, img2 = random.sample(images, 2)
            
            img_path1 = os.path.join(person_dir, img1)
            img_path2 = os.path.join(person_dir, img2)
            
            try:
                # 预处理图像
                image1 = preprocess_image(img_path1, visualize=visualize)
                image2 = preprocess_image(img_path2, visualize=visualize)
                
                # 提取特征
                features1 = extract_features(model, image1)
                features2 = extract_features(model, image2)
                
                # 计算相似度
                similarity = cosine_similarity(features1, features2)
                same_person_similarities.append(similarity)
                
                # 判断是否匹配
                if similarity > threshold:
                    true_positives += 1  # 正确预测为同一个人
                else:
                    false_negatives += 1  # 错误预测为不同人
                
                same_person_pairs += 1
                if same_person_pairs % 10 == 0:
                    print(f'已处理 {same_person_pairs} 对相同人的图像')
            except Exception as e:
                print(f'处理图像时出错: {e}')
    
    # 2. 生成不同人的图像对
    while diff_person_pairs < target_pairs_each and len(person_ids) >= 2:
        # 随机选择两个不同的人
        person_id1, person_id2 = random.sample(person_ids, 2)
        
        person_dir1 = os.path.join(valid_dir, person_id1)
        person_dir2 = os.path.join(valid_dir, person_id2)
        
        # 获取两个人的图像
        images1 = [f for f in os.listdir(person_dir1) if f.endswith('.tiff')]
        images2 = [f for f in os.listdir(person_dir2) if f.endswith('.tiff')]
        
        # 如果两个人都有图像，则可以形成一对
        if images1 and images2:
            # 随机选择每个人的一张图像
            img1 = random.choice(images1)
            img2 = random.choice(images2)
            
            img_path1 = os.path.join(person_dir1, img1)
            img_path2 = os.path.join(person_dir2, img2)
            
            try:
                # 预处理图像
                image1 = preprocess_image(img_path1, visualize=visualize)
                image2 = preprocess_image(img_path2, visualize=visualize)
                
                # 提取特征
                features1 = extract_features(model, image1)
                features2 = extract_features(model, image2)
                
                # 计算相似度
                similarity = cosine_similarity(features1, features2)
                diff_person_similarities.append(similarity)
                
                # 判断是否匹配
                if similarity > threshold:
                    false_positives += 1  # 错误预测为同一个人
                else:
                    true_negatives += 1  # 正确预测为不同人
                
                diff_person_pairs += 1
                if diff_person_pairs % 10 == 0:
                    print(f'已处理 {diff_person_pairs} 对不同人的图像')
            except Exception as e:
                print(f'处理图像时出错: {e}')
    
    # 计算总处理时间
    elapsed_time = time.time() - start_time
    print(f'批量比较完成，总共处理了 {same_person_pairs + diff_person_pairs} 对图像，耗时 {elapsed_time:.2f} 秒')
    
    # 计算评估指标
    total_predictions = true_positives + false_positives + true_negatives + false_negatives
    accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算平均相似度
    avg_same_similarity = sum(same_person_similarities) / len(same_person_similarities) if same_person_similarities else 0
    avg_diff_similarity = sum(diff_person_similarities) / len(diff_person_similarities) if diff_person_similarities else 0
    
    # 打印评估结果
    print('\n评估结果:')
    print(f'阈值: {threshold}')
    print(f'总样本数: {total_predictions}')
    print(f'真阳性 (TP): {true_positives}')
    print(f'假阳性 (FP): {false_positives}')
    print(f'真阴性 (TN): {true_negatives}')
    print(f'假阴性 (FN): {false_negatives}')
    print(f'准确率 (Accuracy): {accuracy:.4f}')
    print(f'精确率 (Precision): {precision:.4f}')
    print(f'召回率 (Recall): {recall:.4f}')
    print(f'F1分数: {f1_score:.4f}')
    print(f'相同人平均相似度: {avg_same_similarity:.4f}')
    print(f'不同人平均相似度: {avg_diff_similarity:.4f}')
    
    # 返回评估指标
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'avg_same_similarity': avg_same_similarity,
        'avg_diff_similarity': avg_diff_similarity
    }


if __name__ == '__main__':
    SingleCompare()
    # 设置参数
    num_pairs = 500  # 要比较的图像对数量
    threshold = 0.75  # 相似度阈值
    visualize = False  # 是否可视化ROI提取过程
    
    # 运行批量比较
    BatchCompare(num_pairs=num_pairs, threshold=threshold, visualize=visualize)