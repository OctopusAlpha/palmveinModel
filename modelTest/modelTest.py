import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import models
from torchvision.models import ResNet18_Weights

class ResNet18Embedder(nn.Module):
    def __init__(self, embedding_size=128):
        super(ResNet18Embedder, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 保留原有的 conv1 权重（3 通道）用于转换
        orig_conv1_weight = self.base_model.conv1.weight.data.clone()
        # 修改第一层卷积，使其接受单通道输入
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 用原 conv1 权重平均后初始化新 conv1
        self.base_model.conv1.weight.data = orig_conv1_weight.mean(dim=1, keepdim=True)
        self.base_model.fc = nn.Linear(512, embedding_size)

    def forward(self, x):
        return self.base_model(x)

def process_image(image_path, size=(224, 224)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    image_resized = cv2.resize(image, size)
    tensor = torch.from_numpy(image_resized).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)  # 输出形状 [1, 1, 224, 224]

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型并加载训练时保存的权重
    model = ResNet18Embedder(embedding_size=128).to(device)
    state_dict = torch.load('../trainModel/saved_model/best_model.pth', map_location=device)
    # 这里建议使用 strict=False 加载剩余权重
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 处理图像（单通道）
    signUp_tensor = process_image('signUp_image.pgm').to(device)
    recog_tensor = process_image('recog_image.pgm').to(device)

    with torch.no_grad():
        signUp_features = model(signUp_tensor)
        recog_features = model(recog_tensor)

    cos_sim = F.cosine_similarity(signUp_features, recog_features, dim=1)
    print("\n特征向量维度验证：")
    print(f"注册图像特征维度: {signUp_features.shape}")
    print(f"识别图像特征维度: {recog_features.shape}")
    print(f"\n余弦相似度结果：{cos_sim.item():.4f}")

if __name__ == '__main__':
    test_model()
