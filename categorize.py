import os
import shutil
if __name__ == '__main__':
    # 指定session1文件夹的路径
    folder_path = './palmvein_dataset/session1'
    # 指定session2文件夹的路径
    folder_path = './palmvein_dataset/session2'

    # 获取文件夹中所有文件的文件名
    files = os.listdir(folder_path)

    # 遍历文件名，进行重命名
    for filename in files:
        if filename.endswith('.tiff'):
            # 提取文件名中的数字部分，并转换为整数
            num = int(filename.split('.')[0])
            # 将数字增加6000
            new_num = num + 6000
            # 生成新的文件名
            new_filename = f'{new_num:05d}.tiff'
            # 重命名文件
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

    print('文件重命名完成。')


    # 指定session1和session2文件夹的路径
    session1_path = './palmvein_dataset/session1'
    session2_path = './palmvein_dataset/session2'

    # 创建一个新的目录来存放新文件夹
    new_directory_path = './original_dataset'
    os.makedirs(new_directory_path, exist_ok=True)

    # 获取两个session文件夹中所有文件的文件名，并排序
    session1_files = sorted([f for f in os.listdir(session1_path) if f.endswith('.tiff')])
    session2_files = sorted([f for f in os.listdir(session2_path) if f.endswith('.tiff')])

    # 确保文件数量正确
    assert len(session1_files) == 6000 and len(session2_files) == 6000, "文件数量不正确"

    # 每10张图像创建一个新文件夹
    for i in range(0, 6000, 10):
        # 创建新文件夹
        new_folder = os.path.join(new_directory_path, f'{i//10 + 1:03d}')
        os.makedirs(new_folder, exist_ok=True)
        
        # 复制session1的图像到新文件夹
        for file in session1_files[i:i+10]:
            shutil.copy(os.path.join(session1_path, file), new_folder)
        
        # 复制session2的图像到新文件夹
        for file in session2_files[i:i+10]:
            shutil.copy(os.path.join(session2_path, file), new_folder)

    print('图像分组完成。')