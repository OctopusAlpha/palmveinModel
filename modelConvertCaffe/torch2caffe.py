import torch
import torchvision.models as models
from torch import nn
from pytorch2caffe import pytorch2caffe

# 定义特征提取模型
class ResNet18Embedder(nn.Module):
    def __init__(self, embedding_size=128):
        super(ResNet18Embedder, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        # 修改输入通道为 1（灰度图）
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后的全连接层为嵌入层
        self.base_model.fc = nn.Linear(512, embedding_size)

    def forward(self, x):
        return self.base_model(x)

# 创建特征提取模型实例
embedding_size = 128
model = ResNet18Embedder(embedding_size=embedding_size)
model.eval()

# 加载训练好的权重（如果有的话）
model_path = '../trainModel/saved_model/best_model.pth'
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

# 创建虚拟输入（灰度图像，通道为 1）
dummy_input = torch.ones([1, 1, 224, 224])  # Batch size 1, 1 channel, 224x224

# Caffe 模型名称和保存路径
name = 'palmvein_model'
save_path = './'

# 转换模型
pytorch2caffe.trans_net(model, dummy_input, name)

# 保存为 Caffe 模型文件
pytorch2caffe.save_prototxt(f'{save_path}{name}.prototxt')
pytorch2caffe.save_caffemodel(f'{save_path}{name}.caffemodel')

print(f'Caffe 模型已保存为 {save_path}{name}.prototxt 和 {save_path}{name}.caffemodel')
