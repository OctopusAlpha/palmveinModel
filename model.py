import torch
import torch.nn as nn
from torchvision import models

class PalmVeinNet(nn.Module):
    def __init__(self, feature_dim=512, pretrained=True):
        """
        初始化掌静脉识别模型
        Args:
            feature_dim (int): 特征向量的维度
            pretrained (bool): 是否使用预训练权重
        """
        super(PalmVeinNet, self).__init__()
        
        # 加载预训练的ResNet18模型
        resnet = models.resnet18(pretrained=pretrained)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 添加新的特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入图像张量
        Returns:
            torch.Tensor: 特征向量
        """
        # 提取特征
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.feature_layer(x)
        
        # 手动实现L2标准化
        norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        x = x / (norm + 1e-10)
        return x
    
    def extract_features(self, x):
        """
        提取输入图像的特征向量
        Args:
            x (torch.Tensor): 输入图像张量
        Returns:
            torch.Tensor: 特征向量
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)