import struct

def print_binary_file_content(filename, feature_dim):
    """
    打印二进制文件中的用户ID和特征向量
    :param filename: 二进制文件路径
    :param feature_dim: 特征向量的维度
    """
    try:
        with open(filename, "rb") as f:
            # 每个用户占用的字节数：4 (用户ID) + 4 * feature_dim (特征向量)
            user_size = 4 + 4 * feature_dim
            # 循环读取每个用户的数据
            while True:
                user_data = f.read(user_size)
                if not user_data:
                    break  # 文件结束

                # 解析用户ID
                user_id_bytes = user_data[:4]
                user_id = struct.unpack("i", user_id_bytes)[0]

                # 解析特征向量
                features_bytes = user_data[4:]
                num_features = feature_dim
                # 确保字节数匹配
                if len(features_bytes) != num_features * 4:
                    print("警告：特征向量字节数不匹配，跳过当前用户")
                    continue
                # 解析特征向量
                features = struct.unpack(f"{num_features}i", features_bytes)

                # 打印结果
                print(f"User ID: {user_id}")
                print("特征向量:", features)
                print()

    except FileNotFoundError:
        print(f"文件 {filename} 未找到")
    except struct.error as e:
        print(f"解析数据时出错：{e}")
    except Exception as e:
        print(f"意外错误：{e}")

if __name__ == "__main__":
    # 示例用法
    filename = "user_features.bin"
    FEATURE_DIM = 128
    print_binary_file_content(filename, FEATURE_DIM)