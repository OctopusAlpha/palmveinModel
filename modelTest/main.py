import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
import cv2
import numpy as np

# 定义特征提取模型
class ResNet18Embedder(nn.Module):
    def __init__(self, embedding_size=128):
        super(ResNet18Embedder, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(512, embedding_size)

    def forward(self, x):
        return self.base_model(x)

# 加载模型
model = ResNet18Embedder(embedding_size=128)
state_dict = torch.load('../best_model.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

def process_image(image_path):
    """图像预处理流水线"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    resized = cv2.resize(image, (224, 224))
    tensor = torch.from_numpy(resized).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]

# 处理两个图像
signUp_tensor = process_image('signUp_image.pgm')
recog_tensor = process_image('recog_image.pgm')

# 提取特征
with torch.no_grad():
    signUp_features = model(signUp_tensor)
    recog_features = model(recog_tensor)

# 计算余弦相似度
cos_sim = F.cosine_similarity(signUp_features, recog_features, dim=1)

print(f"\n特征向量维度验证：")
print(f"注册特征维度: {signUp_features.shape}")
print(f"识别特征维度: {recog_features.shape}")
print(f"\n余弦相似度结果：{cos_sim.item():.4f}")

# 相似度阈值判断示例
threshold = 0.7
print(f"\n相似度 {'大于' if cos_sim > threshold else '小于'} 阈值 {threshold}：{cos_sim.item():.4f}")