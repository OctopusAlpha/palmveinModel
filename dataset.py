import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from preprocess import preprocess_image, ROIExtractor

class PalmVeinDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True, apply_preprocess=True):
        """
        初始化掌静脉数据集
        Args:
            data_dir (str): 数据集根目录，每个子文件夹代表一个类别
            transform (callable, optional): 数据增强和预处理操作
            is_train (bool): 是否为训练模式
            apply_preprocess (bool): 是否应用关键点定位和ROI提取预处理
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.apply_preprocess = apply_preprocess
        
        # 如果需要预处理，创建ROI提取器
        if self.apply_preprocess:
            self.roi_extractor = ROIExtractor()
        
        # 获取所有类别（文件夹名）
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 收集所有图像路径和对应标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith('.tif') or img_name.lower().endswith('.tiff'):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # 设置数据预处理和增强
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
            
            # 如果是训练模式，添加数据增强
            if is_train:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    self.transform
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        """
        返回数据集大小
        Returns:
            int: 数据集中的样本数量
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        Args:
            idx (int): 样本索引
        Returns:
            tuple: (图像张量, 类别标签)
        """
        img_path, label = self.samples[idx]
        
        try:
            # 应用关键点定位和ROI提取预处理
            if self.apply_preprocess:
                try:
                    image = preprocess_image(img_path, self.roi_extractor)
                except Exception as e:
                    print(f"预处理失败 {img_path}: {str(e)}，使用原始图像")
                    # 预处理失败时使用原始图像
                    image = Image.open(img_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
            else:
                # 使用PIL加载TIFF图像
                image = Image.open(img_path)
                # 确保图像是RGB格式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            
            # 应用数据预处理和增强
            if self.transform is not None:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # 返回一个占位图像和标签
            return torch.zeros((3, 224, 224)), label