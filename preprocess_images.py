import torch
from torchvision import transforms
from PIL import Image
import os
import config
from CLAHE import enhance_image
from tqdm import tqdm
import numpy as np

# 定义数据转换（与prepare_data.py中一致）
transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计值
                         std=[0.229, 0.224, 0.225])
])

def preprocess_and_save(source_dir, target_dir):
    """
    预处理图像并保存为 .pt 文件。

    Args:
        source_dir (str): 原始图像所在的根目录。
        target_dir (str): 保存预处理后张量的目标根目录。
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f'找到 {len(classes)} 个类别在 {source_dir}')

    for cls in tqdm(classes, desc=f'处理目录 {os.path.basename(source_dir)}'):
        source_class_dir = os.path.join(source_dir, cls)
        target_class_dir = os.path.join(target_dir, cls)
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        image_files = [f for f in os.listdir(source_class_dir) if os.path.isfile(os.path.join(source_class_dir, f))]
        
        for img_name in tqdm(image_files, desc=f'类别 {cls}', leave=False):
            source_img_path = os.path.join(source_class_dir, img_name)
            # 构建目标文件名，将扩展名替换为 .pt
            base_name, _ = os.path.splitext(img_name)
            target_tensor_path = os.path.join(target_class_dir, f'{base_name}.pt')

            # 如果目标文件已存在，则跳过
            if os.path.exists(target_tensor_path):
                # print(f'跳过已存在的文件: {target_tensor_path}')
                continue

            try:
                # 加载灰度图像
                img = Image.open(source_img_path).convert('L')
                
                # 应用CLAHE增强
                img = enhance_image(img)
                
                # 将灰度图像转换为三通道
                img = Image.merge('RGB', (img, img, img))

                # 应用转换
                if transform:
                    tensor = transform(img)
                else:
                    # 如果没有transform，至少转换为Tensor
                    tensor = transforms.ToTensor()(img)
                
                # 保存张量
                torch.save(tensor, target_tensor_path)

            except Exception as e:
                print(f'处理文件 {source_img_path} 时出错: {e}')

if __name__ == '__main__':
    # 定义预处理后的数据保存目录 (可以移到 config.py)
    preprocessed_train_dir = os.path.join('data', 'preprocessed_train')
    preprocessed_valid_dir = os.path.join('data', 'preprocessed_valid')

    print('开始预处理训练数据...')
    preprocess_and_save(config.train_dir, preprocessed_train_dir)
    print('训练数据预处理完成。')

    print('\n开始预处理验证数据...')
    preprocess_and_save(config.valid_dir, preprocessed_valid_dir)
    print('验证数据预处理完成。')

    print(f'\n预处理后的训练数据保存在: {os.path.abspath(preprocessed_train_dir)}')
    print(f'预处理后的验证数据保存在: {os.path.abspath(preprocessed_valid_dir)}')
    print('\n请确保更新 prepare_data.py 中的 TripletDataset 以使用这些预处理数据。')