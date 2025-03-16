import os
import torch
import numpy as np
from model import PalmVeinNet
from pytorch2caffe import pytorch2caffe

def convert_model(pytorch_model, save_dir):
    """转换完整模型"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保模型处于评估模式
    pytorch_model.eval()
    
    # 准备输入数据
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # 设置输出文件路径
    name = 'palm_vein_net'
    base_path = os.path.join(save_dir, name)
    
    print(f'开始转换模型...')
    
    try:
        # 使用pytorch2caffe进行模型转换
        pytorch2caffe.trans_net(pytorch_model, input_tensor, name)
        
        # 保存为Caffe模型文件
        pytorch2caffe.save_prototxt(f'{base_path}.prototxt')
        pytorch2caffe.save_caffemodel(f'{base_path}.caffemodel')
        
        # 验证文件是否成功生成
        prototxt_path = f'{base_path}.prototxt'
        caffemodel_path = f'{base_path}.caffemodel'
        
        if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
            print(f'模型转换成功！')
            print(f'生成的文件:')
            print(f'- {prototxt_path}')
            print(f'- {caffemodel_path}')
            print(f'文件大小:')
            print(f'- prototxt: {os.path.getsize(prototxt_path) / 1024:.2f} KB')
            print(f'- caffemodel: {os.path.getsize(caffemodel_path) / 1024:.2f} KB')
        else:
            print('警告：模型文件未能正确生成')
            if not os.path.exists(prototxt_path):
                print(f'- 未找到prototxt文件: {prototxt_path}')
            if not os.path.exists(caffemodel_path):
                print(f'- 未找到caffemodel文件: {caffemodel_path}')
            
            # 列出目标目录中的所有文件
            print('\n当前目录下的文件：')
            for file in os.listdir(save_dir):
                print(f'- {file}')
    except Exception as e:
        print(f'模型转换过程中出现错误：{str(e)}')
        raise

def main():
    # 加载PyTorch模型
    model = PalmVeinNet(feature_dim=512)
    
    # 如果有预训练权重，加载它们
    checkpoint_dir = 'checkpoints'
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if model_files:
        latest_model = max([os.path.join(checkpoint_dir, f) for f in model_files], key=os.path.getmtime)
        checkpoint = torch.load(latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'加载模型权重: {latest_model}')
    
    # 执行转换
    save_dir = 'caffe_model'
    convert_model(model, save_dir)

if __name__ == '__main__':
    main()