import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from src import config
from src.core.enhance import enhance_image
from collections import defaultdict

class PalmVeinDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            img_names = [img for img in os.listdir(cls_dir) 
                         if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
            for img_name in img_names:
                self.images.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def _load_image(self, path):
        # Load image
        img = Image.open(path).convert('L')
        # Apply CLAHE (Enhancement)
        img = enhance_image(img)
        # Convert to RGB (3 channels) as ResNet expects 3 channels
        img = Image.merge('RGB', (img, img, img))
        return img

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]
        
        img = self._load_image(path)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

class BalancedBatchSampler(Sampler):
    """
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = np.array(dataset.labels)
        self.labels_set = list(set(self.labels.tolist()))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_classes * self.n_samples

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            # Removed RandomHorizontalFlip as palm veins are chiral/directional
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            # Add RandomAffine for more robust geometric invariance
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders():
    train_transform = get_transforms(is_train=True)
    valid_transform = get_transforms(is_train=False)
    
    train_dataset = PalmVeinDataset(config.train_dir, transform=train_transform, is_train=True)
    valid_dataset = PalmVeinDataset(config.valid_dir, transform=valid_transform, is_train=False)
    
    # P-K Sampling: P classes, K images per class
    # Batch size = P * K.
    # Let's aim for batch size around 32-64.
    # If we have 10 images per person (8 train), we can do K=4, P=8 -> Batch=32
    # Or K=4, P=16 -> Batch=64.
    
    n_classes = 16
    n_samples = 4
    
    train_sampler = BalancedBatchSampler(train_dataset, n_classes=n_classes, n_samples=n_samples)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)
    
    # Validation loader doesn't need balanced sampler, just standard shuffle=False
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, valid_loader
