import numpy as np
import pickle
import glob
import os
# 定义要合并的 .npz 文件路径
from concurrent.futures import ThreadPoolExecutor

# ori_folder = "/ssd1/lishujia/chois_release/mesh_cond_data1/train"
# target_folder = "/ssd1/lishujia/chois_release/mesh_cond_data2/train"
# output_file = os.path.join(target_folder, "merged_data")
# # 创建一个空字典来存储所有数据
# def load_npz(file_path):
#     """ 逐步加载 .npz 并写入 pickle """
#     index = int(os.path.basename(file_path).split('.')[0])
#     print(index)
#     with np.load(file_path, allow_pickle=True) as data:
#         return index, {key: data[key] for key in data.files}

# npz_files = [os.path.join(ori_folder, file) for file in os.listdir(ori_folder) if file.endswith(".npz")]

# batch_size = 1000  # 每个 .p 文件最多存 1000 个数据
# file_count = 0
# merged_data = {}

# with ThreadPoolExecutor(max_workers=16) as executor:
#     for i, (index, data) in enumerate(executor.map(load_npz, npz_files)):
#         merged_data[index] = data
#         if (i + 1) % batch_size == 0:
#             with open(f"{output_file}_{file_count}.p", 'wb') as f:
#                 pickle.dump(merged_data, f)
#             merged_data = {}  # 清空，开始新的 batch
#             file_count += 1

# # **最后一批**
# if merged_data:
#     with open(f"{output_file}_{file_count}.p", 'wb') as f:
#         pickle.dump(merged_data, f)

# input_folder = "/ssd1/lishujia/chois_release/mesh_cond_data2/train"  # 存放 .p 文件的文件夹
# output_file = os.path.join(input_folder, "final_merged_data.p")  # 最终合并后的文件

# merged_data = {}

# # 获取所有 .p 文件，并按文件名排序（防止乱序）
# p_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".p")])

# # 遍历所有 .p 文件，逐个加载并合并
# for p_file in p_files:
#     with open(p_file, 'rb') as f:
#         data = pickle.load(f)
#         merged_data.update(data)  # **合并字典**

#     print(f"Loaded {p_file}, total keys: {len(merged_data)}")  # 显示当前已合并的 key 数量

# # **保存最终合并的字典**
# with open(output_file, 'wb') as f:
#     pickle.dump(merged_data, f)
import os
import numpy as np
import multiprocessing

import os
import numpy as np
import pickle
import multiprocessing

def load_npy(file_path):
    """
    读取 .npy 文件并返回 (键, 数据) 对。
    
    Args:
        file_path: str, .npy 文件路径
        
    Returns:
        tuple: (int, np.ndarray) - 文件名的数值部分作为键，数组作为值
    """
    try:
        key = int(os.path.basename(file_path).split('.')[0])  # 提取文件名前的数字
        print(key)
        data = np.load(file_path, allow_pickle=True)  # 读取 .npy 文件
        return key, data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        #return None  # 遇到错误返回 None，后续会过滤掉

def save_dict_to_pickle(data_dict, output_file):
    """
    保存字典到 .p (pickle) 文件。
    
    Args:
        data_dict: dict, 需要保存的字典
        output_file: str, 输出 .p 文件路径
    """
    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"✅ 数据已保存到 {output_file}")

def process_folder(input_folder, output_file, num_workers=8):
    """
    读取目录下的所有 .npy 文件，并保存为 .p 文件。
    
    Args: 
        input_folder: str, 输入文件夹路径
        output_file: str, 输出 .p 文件路径
        num_workers: int, 进程数量
    """
    # 获取所有 .npy 文件
    npy_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npy')]

    # 多进程读取 .npy 文件
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(load_npy, npy_files)

    # 过滤掉加载失败的项（None）
    data_dict = {k: v for k, v in results if k is not None}

    # 保存到 .p 文件
    save_dict_to_pickle(data_dict, output_file)


if __name__ == "__main__":
    #data = np.load('/ssd1/lishujia/chois_release/mesh_cond_data2/test/00002.npz')
    #data1 = np.load('/ssd1/lishujia/chois_release/mesh_cond_data1/test/00002.npz')
    input_folder = "/ssd1/lishujia/chois_release/mesh_cond_data2/train"  # 替换为你的输入文件夹路径
    output_file = "/ssd1/lishujia/chois_release/mesh_cond_data2/train_data.p"  # 替换为你的输出文件夹路径
 
    process_folder(input_folder, output_file, num_workers=12)
    print("🎉 所有 .npy 文件已处理完毕并保存！")