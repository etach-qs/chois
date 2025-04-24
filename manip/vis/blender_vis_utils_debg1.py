import sys
import os
import bpy
import argparse
import math

def setup_material(obj, color):
    mat = bpy.data.materials.new(name="Material")
    mat.use_nodes = True
    principled_bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs[0].default_value = (*color, 1.0)  # RGBA
    obj.data.materials.append(mat)

def import_and_setup_object(filepath, name, color):
    bpy.ops.import_mesh.ply(filepath=filepath)
    obj = bpy.data.objects.get(name)
    if obj:
        for f in obj.data.polygons:
            f.use_smooth = True
        obj.rotation_euler = (0, 0, 0)
        setup_material(obj, color)
    return obj

if __name__ == "__main__":
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description='Render first frame in Blender.')
    parser.add_argument('--folder', type=str, required=True, help='Path to folder containing .ply files')
    parser.add_argument('--out-folder', type=str, required=True, help='Path to output folder')
    parser.add_argument('--scene', type=str, required=True, help='Path to .blend scene file')
    args = parser.parse_args(argv)

    # Load Blender Scene
    bpy.ops.wm.open_mainfile(filepath=args.scene)
    
    obj_folder = args.folder
    output_dir = args.out_folder
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the first frame
    obj_files = sorted([f for f in os.listdir(obj_folder) if f.endswith(".ply") and "object" not in f])
    if not obj_files:
        print("No PLY files found!")
        sys.exit(1)
    
    first_frame = obj_files[0]
    human_path = os.path.join(obj_folder, first_frame)
    object_path = human_path.replace(".ply", "_object.ply")
    
    # Import and setup first frame
    #import_and_setup_object(human_path, first_frame.replace(".ply", ""), (0.833, 0.417, 0.167))
    import_and_setup_object(object_path, first_frame.replace(".ply", "") + "_object", (53/255, 51/255, 255/255))

    # Render and save output
    bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, "first_frame_render.jpg")
    bpy.ops.render.render(write_still=True)
    
    # Cleanup and exit
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.wm.quit_blender()
