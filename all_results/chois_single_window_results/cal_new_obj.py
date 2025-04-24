import os
import json
import numpy as np
from collections import defaultdict

# 假设 name_list 包含文件名中的关键词
name_list = ["smalltable", "whitechair", "suitcase", "tripod"]
name_list1 = ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", "trashcan", "monitor", \
                    "floorlamp", "clothesstand"]

# 文件夹路径
folder_path = '/ssd1/lishujia/chois_release/chois_result_split_nomesh1/evaluation_metrics_json/chois'

# 输出文件路径
output_file_path = '/ssd1/lishujia/chois_release/chois_result_split_nomesh1/averages_unseen.json'

# 初始化一个字典来存储每个文件的平均值
key_averages = defaultdict(list)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件名是否包含 name_list 中的元素
    if any(name in filename for name in name_list) and filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 将每个键的值添加到 key_averages 中
        for key, values in data.items():
            if isinstance(values, list):  # 确保值是列表
                key_averages[key].extend(values)
            else:
                key_averages[key].append(values)  # 如果值不是列表，直接添加到列表中

# 计算每个键的均值
result = {key: np.mean(values) for key, values in key_averages.items()}

# 将结果写入新的 JSON 文件
with open(output_file_path, 'w') as f:
    json.dump(result, f, indent=4)  # indent=4 用于美化输出格式

print(f"结果已保存到 {output_file_path}")