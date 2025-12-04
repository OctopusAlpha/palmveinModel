import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from src import config
from src.core.enhance import enhance_image

class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_images = {
            cls: [os.path.join(root_dir, cls, img) 
                  for img in os.listdir(os.path.join(root_dir, cls)) 
                  if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            for cls in self.classes
        }
        
        # Filter classes with < 2 images
        self.class_to_images = {cls: imgs for cls, imgs in self.class_to_images.items() if len(imgs) >= 2}
        self.classes = list(self.class_to_images.keys())
        
        if not self.classes:
            raise RuntimeError(f"No valid classes found in {root_dir}")
            
        # Create a flat list of all images for indexing
        self.images = []
        for cls in self.classes:
            for img_path in self.class_to_images[cls]:
                self.images.append((img_path, cls))

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
        # Anchor
        anchor_path, anchor_class = self.images[idx]
        anchor_img = self._load_image(anchor_path)
        
        # Positive (Same class, different image)
        # Get all images for this class
        class_images = self.class_to_images[anchor_class]
        # Exclude the anchor image itself if possible (though if only 2 images, we must pick the other one)
        # If there are duplicates in list, we might pick same image.
        # Let's filter by path.
        possible_positives = [p for p in class_images if p != anchor_path]
        
        if not possible_positives:
            # Should not happen due to len check in init, but fallback:
            positive_path = anchor_path
        else:
            positive_path = random.choice(possible_positives)
            
        positive_img = self._load_image(positive_path)
        
        # Negative (Different class)
        negative_class = random.choice([c for c in self.classes if c != anchor_class])
        negative_path = random.choice(self.class_to_images[negative_class])
        negative_img = self._load_image(negative_path)
        
        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        # Return labels too for potential CenterLoss usage or metrics
        # We need to map class name to integer
        class_idx = self.classes.index(anchor_class)
        
        return anchor_img, positive_img, negative_img, class_idx

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    
    train_dataset = TripletDataset(config.train_dir, transform=train_transform, is_train=True)
    valid_dataset = TripletDataset(config.valid_dir, transform=valid_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, valid_loader
