
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import config

# 定义数据转换
transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),               # 随机旋转
    transforms.RandomHorizontalFlip(),                     # 随机水平翻转
    transforms.RandomVerticalFlip(),                       # 随机垂直翻转
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),# 随机平移
    transforms.RandomResizedCrop(224),                     # 随机裁剪并缩放到224×224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        # 随机选择一个类作为anchor和positive
        anchor_class = random.choice(self.classes)
        anchor_images = self.class_to_images[anchor_class]
        anchor_img_path, positive_img_path = random.sample(anchor_images, 2)

        # 随机选择一个不同的类作为negative
        negative_class = random.choice([cls for cls in self.classes if cls != anchor_class])
        negative_img_path = random.choice(self.class_to_images[negative_class])

        # 加载图像并应用变换
        anchor_img = Image.open(anchor_img_path).convert('RGB')  # 确保为 RGB 图像
        positive_img = Image.open(positive_img_path).convert('RGB')
        negative_img = Image.open(negative_img_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# 获取三元组数据集
def get_triplet_datasets():
    train_dataset = TripletDataset(root_dir=config.train_dir, transform=transform)
    valid_dataset = TripletDataset(root_dir=config.valid_dir, transform=transform)
    test_dataset = TripletDataset(root_dir=config.test_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader
