import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

# 读取 PLY 文件
def load_ply(filepath):
    mesh = trimesh.load(filepath)
    return mesh

# 远点采样 (Farthest Point Sampling)
def farthest_point_sampling(points, num_samples):
    """
    对点云进行远点采样
    """
    n = len(points)
    sampled_indices = [np.random.randint(n)]  # 随机选择一个起始点
    distances = np.linalg.norm(points - points[sampled_indices[0]], axis=1)

    for _ in range(num_samples - 1):
        farthest_index = np.argmax(distances)
        sampled_indices.append(farthest_index)
        new_distances = np.linalg.norm(points - points[farthest_index], axis=1)
        distances = np.minimum(distances, new_distances)

    return points[sampled_indices]

# 可视化点云并保存为图片
def plot_point_cloud(human_vertices, object_vertices, closest_human_vertices, output_image_path):
    """
    使用 matplotlib 绘制点云并保存为图片
    """
    fig = plt.figure(figsize=(5, 30))  # 设置图像尺寸为 10x2 英寸，长宽比为 5:1
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    # 设置背景为白色
    fig.patch.set_facecolor('white')

    # 绘制其他人体点云（绿色，透明度 1）
    other_human_vertices = np.array([v for v in human_vertices if v not in closest_human_vertices])
    ax.scatter(
        other_human_vertices[:, 0], other_human_vertices[:, 1], other_human_vertices[:, 2],
        c=[(0.2, 0.2, 0.2)], label='Other Human', s=0.5, alpha=1
    )

    # 绘制最近的人体点云（黄色，透明度 1）
    ax.scatter(
        closest_human_vertices[:, 0], closest_human_vertices[:, 1], closest_human_vertices[:, 2],
        c=[(0.2, 0.2, 0.2)], label='Closest Human', s=0.5, alpha=1
    )

    # 绘制物体点云（浅蓝色，透明度 1）
    ax.scatter(
        object_vertices[:, 0], object_vertices[:, 1], object_vertices[:, 2],
        c=[(53/255.0, 51/255.0, 255/255.0)], label='Object', s=0.1, alpha=0.5
    )

    # 隐藏坐标轴
    ax.set_axis_off()

    # 设置图例
    #ax.legend()

    # 调整相机视角
    ax.view_init(elev=0, azim=0)

    # 保存图片
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight')
    plt.close()

# 使用 KDTree 计算人体顶点到物体表面的最近点
def calculate_closest_points_kdtree(human_vertices, object_vertices, top_k=3000):
    """
    使用 KDTree 找到 `human_vertices` 中距离 `object_vertices` 最近的 top_k 个点
    """
    tree = KDTree(object_vertices)  # 构建 KDTree
    min_distances, _ = tree.query(human_vertices, k=1)  # 查询每个人体点的最近物体点
    closest_indices = np.argsort(min_distances)[:top_k]  # 选取最近的 top_k 个人体点索引
    return closest_indices

# 主函数
def main():
    # 加载 PLY 文件
    object_mesh = load_ply("/ssd1/lishujia/chois_release/chois_mesh_window_results_inter5_16/objs_single_window_cmp_settings/chois/sub16_largetable_015_largetable_sidx_0_eidx_119_sample_cnt_0/objs_step_10_bs_idx_0/00066_object.ply")
    human_mesh = load_ply("/ssd1/lishujia/chois_release/chois_mesh_window_results_inter5_16/objs_single_window_cmp_settings/chois/sub16_largetable_015_largetable_sidx_0_eidx_119_sample_cnt_0/objs_step_10_bs_idx_0/00066.ply")

    # 获取顶点坐标
    object_vertices = np.array(object_mesh.vertices)
    human_vertices = np.array(human_mesh.vertices)

    # 对物体点云进行远点采样，采集 1000 个点
    object_vertices_sampled = farthest_point_sampling(object_vertices, 500)

    # 缩放所有点云的高度（Z 坐标乘以 1.5）
    object_vertices_sampled[:, 2] *= 1.5
    human_vertices[:, 2] *= 1.5

    # 使用 KDTree 找到距离物体表面最近的 3000 个人体点
    closest_indices = calculate_closest_points_kdtree(human_vertices, object_vertices_sampled, top_k=1000)
    closest_human_vertices = human_vertices[closest_indices]

    # 可视化点云并保存为图片
    plot_point_cloud(human_vertices, object_vertices_sampled, closest_human_vertices, "./point_cloud_plot.png")

# 运行主函数
main()