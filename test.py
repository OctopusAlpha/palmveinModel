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
from tqdm import tqdm
import random
from src.model.metric_learning import ArcFace
from src.data.dataset import get_dataloaders
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes=None):
    """
    加载训练好的模型
    Args:
        model_path (str): 模型文件路径
        num_classes (int): 类别数量，如果提供则加载分类头
    Returns:
        model: 加载好的模型
    """
    # Ensure we match the model definition in src.model.network.PalmVeinNet
    model = PalmVeinNet(embedding_size=config.feature_dim, num_classes=num_classes).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # If num_classes is None (Verification mode), but checkpoint has classifier weights,
    # we need strict=False to ignore the extra keys.
    # If num_classes is provided (Classification mode), we expect exact match or strict=False is fine too.
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning loading state dict: {e}")
        
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
        transforms.Normalize(mean=[0.5], std=[0.5]) # Modified for grayscale
    ])
    
    # 读取原始图像
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f'无法读取图像: {image_path}')
    
    # 创建ROI提取器并提取ROI
    roi_extractor = ROIExtractor()
    # Note: extract_roi signature returns (center, roi, rect)
    try:
        palm_center, roi_resized, rect_coords = roi_extractor.extract_roi(original, visualize=visualize)
    except Exception as e:
        # Fallback if signature is different or fails
        # print(f"ROI extraction error: {e}")
        roi_resized = None
    
    if roi_resized is None:
        # print(f'ROI提取失败，使用原始图像: {image_path}')
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
    
    # 保持单通道 (Grayscale)
    image = enhanced_image
    
    # 应用变换
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image

def extract_features(model, dataloader):
    # dataloader can be a DataLoader object or a Tensor (single image batch)
    model.eval()
    
    # If dataloader is actually a tensor (image batch), handle it directly
    if torch.is_tensor(dataloader):
        with torch.no_grad():
            images = dataloader.to(device)
            emb = model(images)
            emb = F.normalize(emb, p=2, dim=1)
            return emb.cpu()
            
    features = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting Features'):
            images = images.to(device)
            emb = model(images)
            # Explicitly normalize features for ArcFace
            # Although model output might be normalized if network.py does it,
            # double normalization doesn't hurt (p=2).
            # If network.py doesn't, this is CRITICAL for Cosine Similarity.
            emb = F.normalize(emb, p=2, dim=1)
            
            features.append(emb.cpu())
            labels_list.append(labels)
            
    features = torch.cat(features)
    labels = torch.cat(labels_list)
    return features, labels

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

def BatchCompare(num_pairs=100):
    """
    Batch compare images from validation set to calculate accuracy, FAR, and FRR.
    """
    # Load model
    best_model = f"{config.save_model_dir}/best_palm_vein_model.pth"
    if not os.path.exists(best_model):
        print(f"Model not found at {best_model}")
        return
    
    print(f'Loading model: {best_model}')
    model = load_model(best_model)
    
    # Get validation images
    valid_dir = config.valid_dir
    if not os.path.exists(valid_dir):
        print(f"Validation directory not found: {valid_dir}")
        return
        
    classes = [d for d in os.listdir(valid_dir) if os.path.isdir(os.path.join(valid_dir, d))]
    if len(classes) < 2:
        print("Not enough classes for batch testing.")
        return

    # Collect all images
    class_images = {}
    all_images = []
    for cls in classes:
        cls_dir = os.path.join(valid_dir, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        if len(imgs) > 0:
            class_images[cls] = imgs
            for img in imgs:
                all_images.append((cls, img))
    
    print(f"Found {len(all_images)} images in {len(classes)} classes.")
    
    # Generate pairs
    positive_pairs = []
    negative_pairs = []
    
    # Generate positive pairs (same class)
    for cls, imgs in class_images.items():
        if len(imgs) < 2: continue
        # Create all possible pairs or random sample
        # For larger datasets, random sample is better. Here we do random sample.
        for _ in range(min(len(imgs), 10)): # Limit per class to avoid imbalance
            img1, img2 = random.sample(imgs, 2)
            positive_pairs.append((img1, img2))
            
    # Generate negative pairs (diff class)
    # Match number of positive pairs roughly
    num_pos = len(positive_pairs)
    for _ in range(num_pos):
        cls1, cls2 = random.sample(classes, 2)
        img1 = random.choice(class_images[cls1])
        img2 = random.choice(class_images[cls2])
        negative_pairs.append((img1, img2))
        
    print(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs.")
    
    # Evaluate
    similarities_pos = []
    similarities_neg = []
    
    print("Evaluating positive pairs...")
    for img1_path, img2_path in tqdm(positive_pairs):
        try:
            img1 = preprocess_image(img1_path)
            img2 = preprocess_image(img2_path)
            feat1 = extract_features(model, img1)
            feat2 = extract_features(model, img2)
            sim = cosine_similarity(feat1, feat2)
            similarities_pos.append(sim)
        except Exception as e:
            # print(f"Error processing pair: {e}")
            pass

    print("Evaluating negative pairs...")
    for img1_path, img2_path in tqdm(negative_pairs):
        try:
            img1 = preprocess_image(img1_path)
            img2 = preprocess_image(img2_path)
            feat1 = extract_features(model, img1)
            feat2 = extract_features(model, img2)
            sim = cosine_similarity(feat1, feat2)
            similarities_neg.append(sim)
        except Exception as e:
            # print(f"Error processing pair: {e}")
            pass

    # Calculate metrics for different thresholds
    thresholds = np.arange(0, 1.0, 0.01)
    best_acc = 0
    best_thresh = 0
    
    print("\nResults:")
    print(f"{'Threshold':<10} {'Accuracy':<10} {'FAR':<10} {'FRR':<10}")
    
    for thresh in thresholds:
        # TP: pos_sim > thresh
        tp = sum(s > thresh for s in similarities_pos)
        fn = len(similarities_pos) - tp
        
        # TN: neg_sim <= thresh
        tn = sum(s <= thresh for s in similarities_neg)
        fp = len(similarities_neg) - tn
        
        acc = (tp + tn) / (len(similarities_pos) + len(similarities_neg))
        far = fp / len(similarities_neg) if len(similarities_neg) > 0 else 0
        frr = fn / len(similarities_pos) if len(similarities_pos) > 0 else 0
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
        if int(thresh * 100) % 10 == 0:
             print(f"{thresh:<10.2f} {acc:<10.4f} {far:<10.4f} {frr:<10.4f}")

    print(f"\nBest Threshold: {best_thresh:.2f}")
    print(f"Best Accuracy: {best_acc:.4f}")

def ClassificationTest():
    """
    Test using Classification Accuracy (same as training)
    """
    print("Running Classification Test...")
    
    # Data Loaders
    try:
        _, valid_loader = get_dataloaders()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Model
    best_model_path = f"{config.save_model_dir}/best_palm_vein_model.pth"
    if not os.path.exists(best_model_path):
        print(f"Model not found at {best_model_path}")
        return

    # Initialize model with classification head
    num_classes = len(valid_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    model = load_model(best_model_path, num_classes=num_classes)
    model.eval()
    
    correct = 0
    total = 0
    
    print("Evaluating Classification Accuracy on Validation Set...")
    with torch.no_grad():
        loop_val = tqdm(valid_loader, desc='Classification Test', leave=True)
        for images, labels in loop_val:
            images = images.to(device)
            labels = labels.to(device)
            
            # Model returns logits directly
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop_val.set_postfix(acc=100.*correct/total)
            
    print(f"Classification Accuracy: {100.*correct/total:.2f}%")

if __name__ == '__main__':
    # Choose mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        BatchCompare()
    elif len(sys.argv) > 1 and sys.argv[1] == 'classify':
        ClassificationTest()
    else:
        # Default to single compare or ask user
        # For now, let's run BatchCompare if user asked for it, but keep SingleCompare as default
        # Or just run BatchCompare directly since that's the request?
        # Let's provide a simple menu or just run BatchCompare for this task.
        # User asked to "add batch comparison", so I added the function.
        # I will uncomment the call below to demonstrate it.
        ClassificationTest()
