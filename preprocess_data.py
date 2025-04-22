import torch
from torchvision import transforms
from PIL import Image
import os
import config
from CLAHE import enhance_image
from tqdm import tqdm
import numpy as np

# 与 prepare_data.py 中定义的数据转换保持一致
transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计值
                         std=[0.229, 0.224, 0.225])
])

def preprocess_and_save(source_dir, target_dir):
    """加载、预处理图像并将其保存为 .pt 张量文件。"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目录: {target_dir}")

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"在 {source_dir} 中找到 {len(classes)} 个类别。开始预处理...")

    for cls in tqdm(classes, desc=f"处理 {os.path.basename(source_dir)}"):
        source_class_dir = os.path.join(source_dir, cls)
        target_class_dir = os.path.join(target_dir, cls)
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        image_files = [f for f in os.listdir(source_class_dir) if f.lower().endswith(('tiff','.png', '.jpg', '.jpeg', '.bmp', '.gif', '.pgm'))]
        
        for img_name in image_files:
            source_img_path = os.path.join(source_class_dir, img_name)
            # 构建 .pt 文件名
            base_name, _ = os.path.splitext(img_name)
            target_tensor_path = os.path.join(target_class_dir, f"{base_name}.pt")

            # 如果 .pt 文件已存在，则跳过
            if os.path.exists(target_tensor_path):
                continue

            try:
                # 加载灰度图像
                img = Image.open(source_img_path).convert('L')
                # 应用CLAHE增强
                img = enhance_image(img)
                # 将灰度图像转换为三通道
                img = Image.merge('RGB', (img, img, img))
                # 应用变换
                tensor = transform(img)
                # 保存张量
                torch.save(tensor, target_tensor_path)
            except Exception as e:
                print(f"处理文件 {source_img_path} 时出错: {e}")

if __name__ == '__main__':
    print("开始预处理训练数据...")
    preprocess_and_save(config.train_dir, config.preprocessed_train_dir)
    print("训练数据预处理完成。")

    print("\n开始预处理验证数据...")
    preprocess_and_save(config.valid_dir, config.preprocessed_valid_dir)
    print("验证数据预处理完成。")

    print("\n所有数据预处理完毕。")