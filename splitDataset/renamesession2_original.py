import os

# 指定session2文件夹的路径
folder_path = './session2'

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