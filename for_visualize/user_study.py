import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ["GenHOI vs. OMOMO", "GenHOI vs. CHOIS", "GenHOI vs. GenHOI w/o MF"]  # 横轴文本
data = np.array([
    [84, 10, 6],  # GenHOI vs. OMOMO 的三块数据
    [67, 30, 3],  # GenHOI vs. CHOIS 的三块数据
    [91, 6, 3],  # GenHOI vs. GenHOI w/o MF 的三块数据
])

# 颜色
colors = ['#1116e0', '#70c49c', '#f5b271']  # 三种颜色

# 图例标签
prefer_list = ['Prefer GenHOI', 'Prefer Other', 'No Preference']

# 绘制堆叠柱状图
fig, ax = plt.subplots()
bottom = np.zeros(len(categories))  # 底部初始化为0

# 设置柱状图宽度
bar_width = 0.5  # 调整宽度，默认是 0.8

for i, color in enumerate(colors):
    ax.bar(categories, data[:, i], bottom=bottom, color=color, label=prefer_list[i], width=bar_width)
    bottom += data[:, i]  # 更新底部位置

# 添加标签和标题
ax.set_ylabel('Percentage (%)')
ax.set_title('User Preference Comparison')
ax.legend(loc='upper right')

# 保存图片
plt.savefig('./stacked_bar_chart.png', dpi=300, bbox_inches='tight')  # 保存为PNG文件
plt.close()  # 关闭图表，释放内存