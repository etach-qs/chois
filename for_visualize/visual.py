import bpy
import numpy as np

# 读取 PLY 文件
def load_ply(filepath):
    bpy.ops.import_mesh.ply(filepath=filepath)
    return bpy.context.selected_objects[0]

# 使用 NumPy 批量上色
def color_vertices_numpy(obj, indices, color):
    """
    用 NumPy 批量修改 Blender 颜色数据，避免 for 循环
    """
    mesh = obj.data
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    color_layer = mesh.vertex_colors.active.data

    num_verts = len(mesh.vertices)
    colors = np.zeros((num_verts, 4), dtype=np.float32)  # 默认黑色 (0,0,0,1)
    colors[:, 3] = 1.0  # Alpha 设为 1.0
    colors[indices] = np.array(color, dtype=np.float32)  # 给指定索引赋值颜色

    # ​**按 Loop 赋值颜色**
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            color_layer[loop_idx].color = colors[vert_idx]

# 添加材质，启用 Vertex Color 显示
def setup_vertex_color_material(obj):
    mat = bpy.data.materials.new(name="VertexColorMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    
    if bsdf:
        vcol = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")
        vcol.layer_name = obj.data.vertex_colors.active.name
        mat.node_tree.links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
    
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

# 主函数
def main():
    # ​**清空 Blender 场景**
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # ​**加载 PLY 文件**
    object_ply = load_ply("/ssd1/lishujia/chois_release/chois_single_window_results/objs_single_window_cmp_settings/chois/sub16_clothesstand_002_clothesstand_sidx_0_eidx_119_sample_cnt_0/objs_step_10_bs_idx_0_gt/00119_object.ply")
    human_ply = load_ply("/ssd1/lishujia/chois_release/chois_single_window_results/objs_single_window_cmp_settings/chois/sub16_clothesstand_002_clothesstand_sidx_0_eidx_119_sample_cnt_0/objs_step_10_bs_idx_0_gt/00119.ply")

    # ​**获取顶点坐标**
    object_vertices = np.array([v.co for v in object_ply.data.vertices])
    human_vertices = np.array([v.co for v in human_ply.data.vertices])

    # ​**计算物体中心点**
    object_center = np.mean(object_vertices, axis=0)

    # ​**计算人体点到物体中心点的距离**
    distances = np.linalg.norm(human_vertices - object_center, axis=1)

    # ​**找到最近的 3000 个人体点**
    closest_indices = np.argsort(distances)[:3000]

    # ​**人体顶点上色（黄色）​**
    yellow_color = (1.0, 1.0, 0.0, 1.0)  # RGBA
    color_vertices_numpy(human_ply, closest_indices, yellow_color)

    # ​**物体顶点上色（浅蓝色）​**
    light_blue_color = (53/255.0, 51/255.0, 255/255.0, 1.0)  # RGBA
    color_vertices_numpy(object_ply, range(len(object_vertices)), light_blue_color)

    # ​**设置材质，启用 Vertex Color**
    setup_vertex_color_material(object_ply)
    setup_vertex_color_material(human_ply)

    # ​**确保物体和人体都被选中**
    bpy.ops.object.select_all(action='DESELECT')
    object_ply.select_set(True)
    human_ply.select_set(True)
    bpy.context.view_layer.objects.active = object_ply  # 设定一个活跃物体

    # ​**保存 Blender 文件**
    output_blend_path = "./output.blend"
    bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)

# ​**运行主函数**
main()