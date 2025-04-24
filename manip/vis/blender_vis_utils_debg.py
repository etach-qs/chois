import sys
import os
import math
import bpy
import argparse
import math

def color_human_vertices(human_obj, indices, yellow_color=(1.0, 1.0, 0.0, 1.0), black_color=(0.0, 0.0, 0.0, 1.0)):
    """
    为人体点云的指定索引的顶点设置黄色，其他顶点设置黑色
    """
    mesh = human_obj.data
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()  # 创建顶点颜色层
    color_layer = mesh.vertex_colors.active.data

    # 将所有顶点初始化为黑色
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            color_layer[loop_idx].color = black_color

    # 将指定索引的顶点设置为黄色
    for i in indices:
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                if mesh.loops[loop_idx].vertex_index == i:
                    color_layer[loop_idx].color = yellow_color
if __name__ == "__main__":
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment.')
    parser.add_argument('--folder', type=str, metavar='PATH', help='Path to folder containing .obj/.ply files', default='')
    parser.add_argument('--out-folder', type=str, metavar='PATH', help='Path to output folder', default='')
    parser.add_argument('--scene', type=str, metavar='PATH', help='Path to .blend file for 3D scene', default='')
    args = parser.parse_args(argv)

    WORLD_FILE = args.scene
    bpy.ops.wm.open_mainfile(filepath=WORLD_FILE)

# 设置相机位置 (调整 XYZ 轴)
    # camera = bpy.data.objects["Camera"]
    # camera.location = (5, -7.5, 5)  # 你可以根据需求修改

    # # 设置相机旋转角度 (弧度制，使用 math.radians)
    # camera.rotation_euler = (math.radians(60), math.radians(0), math.radians(30))  # 调整 Pitch, Yaw, Roll



    # bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))  # 地板中心放在 (0,0,0)
    # floor = bpy.context.object
    # floor.name = "Floor"

    # # 创建地板材质
    # floor_mat = bpy.data.materials.new(name="FloorMaterial")
    # floor_mat.use_nodes = True

    # # 设定地板颜色 (灰色)
    # principled_bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
    # if principled_bsdf:
    #     principled_bsdf.inputs[0].default_value = (0.0, 0.0, 0.0, 1)  # 设定为灰色

    # # 赋予地板材质
    # floor.data.materials.append(floor_mat)
    # 创建地板对象
    # bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))  # 地板中心放在 (0,0,0)
    # floor = bpy.context.object
    # floor.name = "Floor"

    # # 创建地板材质
    # floor_mat = bpy.data.materials.new(name="FloorMaterial")
    # floor_mat.use_nodes = True

    # # 设定地板颜色 (灰色)
    # principled_bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
    # if principled_bsdf:
    #     principled_bsdf.inputs[0].default_value = (0.0, 0.0, 0.0, 1)  # 设定为灰色

    # # 赋予地板材质
    # floor.data.materials.append(floor_mat)



    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1

    obj_folder = args.folder
    output_dir = args.out_folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    obj_files = sorted([f for f in os.listdir(obj_folder) if f.endswith(".ply") and "object" not in f])
    selected_frames = obj_files[::30] #+ [obj_files[-1]] # Select every 20th frame
    
    for idx, file_name in enumerate(selected_frames):
        path_to_file = os.path.join(obj_folder, file_name)
        object_path_to_file = path_to_file.replace(".ply", "_object.ply")
     
        # human_new_obj = bpy.ops.import_mesh.ply(filepath=path_to_file)
        # human_obj_object = bpy.data.objects[file_name.replace(".ply", "")]
        # human_mesh = human_obj_object.data
        
        # for f in human_mesh.polygons:
        #     f.use_smooth = True
        # human_obj_object.rotation_euler = (0, 0, 0)

        # human_mat = bpy.data.materials.new(name=f"HumanMaterial_{idx}")
        # human_obj_object.data.materials.append(human_mat)
        # human_mat.use_nodes = True
        # principled_bsdf = human_mat.node_tree.nodes['Principled BSDF']
        # if principled_bsdf is not None:
        #     # shade = idx / len(selected_frames)  # Interpolating color from light to dark
        #     # print(shade, shade * 0.5, 1.0 - shade)
        #     # principled_bsdf.inputs[0].default_value = (0.8333333333333334, 0.4166666666666667, 0.16666666666666663,1)
        #     shade = idx / (len(selected_frames) - 1)  # 计算当前索引的插值系数（0 ~ 1）

        #     # 定义颜色范围
        #     color_light = (0.9, 0.6, 0.4)    # 更浅的颜色
        #     color_mid = (0.833, 0.417, 0.167)  # 中间目标颜色
        #     color_dark = (0.6, 0.3, 0.1)      # 更深的颜色

        #     # 根据 shade 进行插值计算
        #     if shade < 0.5:
        #         t = shade * 2  # 归一化到 0~1
        #         color = (
        #             (1 - t) * color_light[0] + t * color_mid[0],
        #             (1 - t) * color_light[1] + t * color_mid[1],
        #             (1 - t) * color_light[2] + t * color_mid[2],
        #             1
        #         )
        #     else:
        #         t = (shade - 0.5) * 2  # 归一化到 0~1
        #         color = (
        #             (1 - t) * color_mid[0] + t * color_dark[0],
        #             (1 - t) * color_mid[1] + t * color_dark[1],
        #             (1 - t) * color_mid[2] + t * color_dark[2],
        #             1
        #         )

        #     # 应用颜色到材质
        #     principled_bsdf.inputs[0].default_value = color
                
          
                    # shade = idx / len(obj_files)  # 根据帧索引计算渐变比例
            # r = 1.0  # 保持红色不变
            # g = 0.8 - shade * 0.4  # 从 0.8 渐变到 0.4
            # b = 0.6 - shade * 0.6  # 从 0.6 渐变到 0.0
            # principled_bsdf.inputs[0].default_value = (r, g, b, 1)  # 设置材质颜色


        #human_obj_object.active_material = human_mat

        new_obj = bpy.ops.import_mesh.ply(filepath=object_path_to_file)
        obj_object = bpy.data.objects[file_name.replace(".ply", "") + "_object"]
        mesh = obj_object.data
        for f in mesh.polygons:
            f.use_smooth = True  #153/255.0, 51/255.0, 255/255.0
        obj_object.rotation_euler = (0, 0, 0)

        mat = bpy.data.materials.new(name="ObjectMaterial")
        obj_object.data.materials.append(mat)
        mat.use_nodes = True
        principled_bsdf = mat.node_tree.nodes['Principled BSDF']
        if principled_bsdf is not None:
            principled_bsdf.inputs[0].default_value = (53/255.0, 51/255.0, 255/255.0, 1)
        obj_object.active_material = mat

    bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, "final_composite.jpg")
    bpy.ops.render.render(write_still=True)

    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.wm.quit_blender()
