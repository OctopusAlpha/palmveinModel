import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import config
import cv2
import numpy as np
from CLAHE import enhance_image

# 定义数据转换（单通道）
transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计值
                         std=[0.229, 0.224, 0.225])
])

# 自定义三元组数据集类
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_images = {
            cls: [os.path.join(root_dir, cls, img) for img in os.listdir(os.path.join(root_dir, cls))]
            for cls in self.classes
        }

    def __len__(self):
        return sum(len(images) for images in self.class_to_images.values())

    def __getitem__(self, idx):
        anchor_class = random.choice(self.classes)
        anchor_images = self.class_to_images[anchor_class]
        anchor_img_path, positive_img_path = random.sample(anchor_images, 2)

        negative_class = random.choice([cls for cls in self.classes if cls != anchor_class])
        negative_img_path = random.choice(self.class_to_images[negative_class])

        # 加载灰度图像
        anchor_img = Image.open(anchor_img_path).convert('L')
        positive_img = Image.open(positive_img_path).convert('L')
        negative_img = Image.open(negative_img_path).convert('L')
        
        # 应用CLAHE增强
        anchor_img = enhance_image(anchor_img)
        positive_img = enhance_image(positive_img)
        negative_img = enhance_image(negative_img)
        
        # 将灰度图像转换为三通道
        anchor_img = Image.merge('RGB', (anchor_img, anchor_img, anchor_img))
        positive_img = Image.merge('RGB', (positive_img, positive_img, positive_img))
        negative_img = Image.merge('RGB', (negative_img, negative_img, negative_img))

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# 获取三元组数据集
def get_triplet_datasets():
    train_dataset = TripletDataset(root_dir=config.train_dir, transform=transform)
    valid_dataset = TripletDataset(root_dir=config.valid_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, valid_dataloader
