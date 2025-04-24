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



def sample_point_cloud(point_cloud, target_points):
    """
    远点采样点云（如果数据较大，建议用 PyTorch 实现更快的 FPS）。
    
    Args:
        point_cloud: np.ndarray, 形状为 [frame, num_points, 3]
        target_points: int, 目标点数
        
    Returns:
        sampled_point_cloud: np.ndarray, 形状为 [frame, target_points, 3]
    """
    frame, num_points, _ = point_cloud.shape
    sampled_indices = np.random.choice(num_points, target_points, replace=False)  # ✅ 修正 axis 错误
    return point_cloud[:, sampled_indices, :]

def process_file(file_path, output_folder):
    """
    处理单个 .npz 文件：
    1. 读取 human_mesh 和 obj_mesh
    2. 分别采样 2000 和 1000 点
    3. 在 axis=1 方向拼接
    4. 以 .npy 格式保存到目标文件夹

    Args:
        file_path: str, 输入 .npz 文件路径
        output_folder: str, 输出文件夹路径
    """
    # try:
    with np.load(file_path, allow_pickle=True) as data:
        human_mesh = data['human_mesh']  # 形状 [frame, 3000, 3]
        obj_mesh = data['obj_mesh']  # 形状 [frame, 1500, 3]

            # 采样
        sampled_human_mesh = sample_point_cloud(human_mesh, target_points=2000)  # [frame, 2000, 3]
        sampled_obj_mesh = sample_point_cloud(obj_mesh, target_points=1000)  # [frame, 1000, 3]

            # 合并
        merged_mesh = np.concatenate((sampled_human_mesh, sampled_obj_mesh), axis=1)  # [frame, 3000, 3]

            # 保存为 .npy
        file_name = os.path.splitext(os.path.basename(file_path))[0] + ".npy"
        output_path = os.path.join(output_folder, file_name)
        np.save(output_path, merged_mesh)

    print(f"✅ Processed: {file_name}")
    # except Exception as e:
    #     print(f"❌ Error processing {file_path}: {e}")

def process_folder(input_folder, output_folder, num_workers=8):
    """
    使用多进程处理文件夹中的 .npz 文件，并保存为 .npy 文件。

    Args:
        input_folder: str, 输入文件夹路径
        output_folder: str, 输出文件夹路径
        num_workers: int, 进程数量
    """
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

    # 获取所有 .npz 文件
    npz_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npz')]

    # 多进程处理
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(process_file, [(file, output_folder) for file in npz_files])




if __name__ == "__main__":

    input_folder = "/ssd1/lishujia/chois_release/mesh_cond_data1/train"  # 修改为你的 .npz 文件夹路径
    output_folder = "/ssd1/lishujia/chois_release/mesh_cond_data2/train"  # 修改为你的输出文件夹路径
    # data1 = np.load('/ssd1/lishujia/chois_release/mesh_cond_data1/test/00000.npz')
    # data2 = np.load('/ssd1/lishujia/chois_release/mesh_cond_data2/train/00000.npy')
    # import pdb; pdb.set_trace()
    process_folder(input_folder, output_folder, num_workers=12)
    print("🎉 所有 .npz 文件已处理并保存为 .npy！")
