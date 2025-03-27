import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import config
from prepare_data import get_triplet_datasets
import time
import os
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 创建改进的特征提取模型
class ResNet18Embedder(nn.Module):
    def __init__(self, embedding_size=128, dropout_rate=0.5):
        super(ResNet18Embedder, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 添加Dropout和Batch Normalization
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn = nn.BatchNorm1d(512)
        
        # 改进全连接层结构
        self.base_model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            self.bn,
            self.dropout,
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        # 提取基础特征
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        # 添加缺失的残差块
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=1)
        return x

model = ResNet18Embedder(embedding_size=128).to(device)

# 定义改进的Triplet Loss和Center Loss的组合
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        self.device = device
        self.feat_dim = feat_dim

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.centers.size(0)) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.centers.size(0), batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.centers.size(0)).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.centers.size(0))
        mask = labels.eq(classes.expand(batch_size, self.centers.size(0)))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = torch.norm(anchor - positive, p=2, dim=1)
        negative_distance = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(positive_distance - negative_distance + self.margin, min=0.0)
        return loss.mean()

criterion = TripletLoss(margin=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 定义Early Stopping类
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=10, min_delta=0.01)
if __name__ == '__main__':
    os.makedirs(config.save_model_dir, exist_ok=True)
    # 加载三元组数据
    train_dataloader, valid_dataloader = get_triplet_datasets()

    best_valid_loss = float('inf')
    learning_rates = []
    
    # 用于可视化的列表
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()  # 记录当前epoch的开始时间
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        # 使用tqdm包装训练数据加载器，显示进度条
        train_loop = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]', leave=True)
        for data in train_loop:
            anchor, positive, negative = data
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # 前向传播
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            # 计算损失
            loss = criterion(anchor_out, positive_out, negative_out)

            # 计算准确率（正样本距离小于负样本距离的比例）
            positive_distance = torch.norm(anchor_out - positive_out, p=2, dim=1)
            negative_distance = torch.norm(anchor_out - negative_out, p=2, dim=1)
            correct_train += torch.sum(positive_distance < negative_distance).item()
            total_train += anchor_out.size(0)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
            # 计算当前批次的准确率
            batch_accuracy = torch.sum(positive_distance < negative_distance).item() / anchor_out.size(0)
            
            # 更新进度条显示当前批次的损失值和准确率
            train_loop.set_postfix(loss=f'{loss.item():.4f}', acc=f'{batch_accuracy:.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.6f}')

        train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = correct_train / total_train

        # 验证阶段
        model.eval()
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            # 使用tqdm包装验证数据加载器，显示进度条
            valid_loop = tqdm(valid_dataloader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Valid]', leave=True)
            for data in valid_loop:
                anchor, positive, negative = data
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)

                loss = criterion(anchor_out, positive_out, negative_out)
                total_valid_loss += loss.item()
                
                # 计算准确率（正样本距离小于负样本距离的比例）
                positive_distance = torch.norm(anchor_out - positive_out, p=2, dim=1)
                negative_distance = torch.norm(anchor_out - negative_out, p=2, dim=1)
                correct_valid += torch.sum(positive_distance < negative_distance).item()
                total_valid += anchor_out.size(0)
                
                # 计算当前批次的准确率
                batch_accuracy = torch.sum(positive_distance < negative_distance).item() / anchor_out.size(0)
                
                # 更新进度条显示当前批次的损失值和准确率
                valid_loop.set_postfix(loss=f'{loss.item():.4f}', acc=f'{batch_accuracy:.4f}')

        valid_loss = total_valid_loss / len(valid_dataloader)
        valid_accuracy = correct_valid / total_valid

        # 调整学习率
        scheduler.step(valid_loss)

        # 保存当前学习率、损失值和准确率用于可视化
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        # 检查Early Stopping
        if early_stopping(valid_loss):
            print("Early stopping triggered")
            break

        epoch_end_time = time.time()  # 记录当前epoch的结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算训练时间

        print("Epoch: {}/{}, train loss: {:.5f}, valid loss: {:.5f}, train acc: {:.5f}, valid acc: {:.5f}, lr: {:.5f}, duration: {:.2f} seconds".format(
            epoch + 1, config.EPOCHS, train_loss, valid_loss, train_accuracy, valid_accuracy, current_lr, epoch_duration))

        # 保存最低验证损失模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{config.save_model_dir}/best_model.pth")
            print(f"最低验证损失模型已保存为 best_model.pth 文件")
        
        # 实时可视化训练过程
        if (epoch + 1) % 5 == 0 or epoch == 0:  # 每5个epoch更新一次图表，或第一个epoch后
            os.makedirs('results', exist_ok=True)
            plt.figure(figsize=(12, 12))
            
            # 绘制损失曲线
            plt.subplot(3, 1, 1)
            plt.plot(train_losses, label='训练损失')
            plt.plot(valid_losses, label='验证损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('损失曲线')
            
            # 绘制准确率曲线
            plt.subplot(3, 1, 2)
            plt.plot(train_accuracies, label='训练准确率')
            plt.plot(valid_accuracies, label='验证准确率')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('准确率曲线')
            
            # 绘制学习率变化曲线
            plt.subplot(3, 1, 3)
            plt.plot(learning_rates)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('学习率变化')
            
            plt.tight_layout()
            plt.savefig('results/training_curves_current.png')
            plt.close()
            
            print(f"Epoch {epoch+1}: 当前训练过程可视化已保存到 results/training_curves_current.png")

    # 最后保存模型
    torch.save(model.state_dict(), f"{config.save_model_dir}/model_final.pth")
    print("最终模型已保存为 model_final.pth 文件")

    # 可视化训练过程
    plt.figure(figsize=(12, 12))
    
    # 绘制损失曲线
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(valid_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('损失曲线')
    
    # 绘制准确率曲线
    plt.subplot(3, 1, 2)
    plt.plot(train_accuracies, label='训练准确率')
    plt.plot(valid_accuracies, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('准确率曲线')
    
    # 绘制学习率变化曲线
    plt.subplot(3, 1, 3)
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('学习率变化')
    
    # 保存图表
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/training_curves.png')
    plt.close()
    
    # 保存训练指标到文件
    with open('results/training_metrics.txt', 'w') as f:
        f.write('Epoch,TrainLoss,ValidLoss,TrainAcc,ValidAcc,LR\n')
        for i in range(len(train_losses)):
            f.write(f'{i+1},{train_losses[i]:.6f},{valid_losses[i]:.6f},{train_accuracies[i]:.6f},{valid_accuracies[i]:.6f},{learning_rates[i]:.6f}\n')
    
    print("训练过程可视化已保存到 results/training_curves.png")
    print("训练指标已保存到 results/training_metrics.txt")
    
    # 保存学习率到文件
    with open(f"{config.save_model_dir}/learning_rates.txt", 'w') as f:
        for lr in learning_rates:
            f.write(f"{lr}\n")
    print("学习率已保存为 learning_rates.txt 文件")

    # 测试模型输出特征向量
    model.eval()
    dummy_input = torch.ones([1, 3, 224, 224]).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print("模型输出特征向量大小:", output.shape)
    print("模型输出特征向量值:", output)