import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import DataLoader, Dataset
import os
import random
import config

# 自定义三元组数据集类 - 加载预处理的张量
class TripletDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_tensors = {
            cls: [os.path.join(root_dir, cls, tensor_file) 
                  for tensor_file in os.listdir(os.path.join(root_dir, cls)) 
                  if tensor_file.endswith('.pt')] # 加载 .pt 文件
            for cls in self.classes
        }
        # 过滤掉少于2个张量的类别
        self.class_to_tensors = {cls: paths for cls, paths in self.class_to_tensors.items() if len(paths) >= 2}
        self.classes = list(self.class_to_tensors.keys())
        if not self.classes:
            raise RuntimeError(f"在 {root_dir} 中没有找到至少包含2个 '.pt' 文件的类别。请确保已运行 preprocess_data.py。")
        print(f"在 {root_dir} 中找到 {len(self.classes)} 个有效类别。")

    def __len__(self):
        # 计算总张量数
        return sum(len(tensors) for tensors in self.class_to_tensors.values())

    def __getitem__(self, idx):
        # 随机选择一个类别作为锚点
        anchor_class = random.choice(self.classes)
        anchor_tensors = self.class_to_tensors[anchor_class]
        
        # 从锚点类别中随机选择两个不同的张量路径作为锚点和正样本
        anchor_tensor_path, positive_tensor_path = random.sample(anchor_tensors, 2)

        # 随机选择一个不同的类别作为负样本类别
        negative_class = random.choice([cls for cls in self.classes if cls != anchor_class])
        negative_tensors = self.class_to_tensors[negative_class]
        # 从负样本类别中随机选择一个张量路径
        negative_tensor_path = random.choice(negative_tensors)

        # 加载张量
        anchor_tensor = torch.load(anchor_tensor_path,weights_only=True)
        positive_tensor = torch.load(positive_tensor_path,weights_only=True)
        negative_tensor = torch.load(negative_tensor_path,weights_only=True)

        return anchor_tensor, positive_tensor, negative_tensor

# 获取三元组数据集 - 使用预处理后的数据
def get_triplet_datasets():
    # 确保预处理目录存在
    if not os.path.exists(config.preprocessed_train_dir) or not os.path.exists(config.preprocessed_valid_dir):
        raise FileNotFoundError("预处理数据目录不存在。请先运行 preprocess_data.py 脚本。")
        
    print(f"从 {config.preprocessed_train_dir} 加载训练数据...")
    train_dataset = TripletDataset(root_dir=config.preprocessed_train_dir)
    print(f"从 {config.preprocessed_valid_dir} 加载验证数据...")
    valid_dataset = TripletDataset(root_dir=config.preprocessed_valid_dir)

    # 检查数据集是否为空
    if len(train_dataset) == 0:
        print(f"警告: 训练数据集为空 ({config.preprocessed_train_dir})。")
    if len(valid_dataset) == 0:
        print(f"警告: 验证数据集为空 ({config.preprocessed_valid_dir})。")

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, valid_dataloader
