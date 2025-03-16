import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from model import PalmVeinNet
from dataset import PalmVeinDataset

def train_model(data_dir, batch_size=64, num_epochs=100, feature_dim=512,
               learning_rate=0.001, save_dir='checkpoints'):
    """
    训练掌静脉识别模型
    Args:
        data_dir (str): 数据集根目录
        batch_size (int): 批次大小
        num_epochs (int): 训练轮数
        feature_dim (int): 特征向量维度
        learning_rate (float): 学习率
        save_dir (str): 模型保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    train_dataset = PalmVeinDataset(data_dir, is_train=True, apply_preprocess=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0, pin_memory=True)
    
    # 获取类别数量
    num_classes = len(train_dataset.classes)
    
    # 创建模型，添加分类层
    model = PalmVeinNet(feature_dim=feature_dim).to(device)
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 检查数据维度
            if images.size(0) != labels.size(0):
                print(f"维度不匹配: images {images.size()}, labels {labels.size()}")
                continue
                
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            features = model(images)
            outputs = classifier(features)
            
            # 检查输出维度
            if outputs.size(1) != num_classes:
                print(f"输出维度错误: {outputs.size()} vs {num_classes}")
                continue
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 打印训练信息
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] '
                      f'Batch [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(save_dir, f'best_model_{timestamp}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f'保存最佳模型到 {save_path}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}] 平均损失: {avg_loss:.4f}')

def main():
    # 设置训练参数
    params = {
        'data_dir': 'dataset/train',  # 训练数据集目录
        'batch_size': 32,
        'num_epochs': 100,
        'feature_dim': 512,
        'learning_rate': 0.001,
        'save_dir': 'checkpoints'
    }
    
    # 开始训练
    train_model(**params)

if __name__ == '__main__':
    main()