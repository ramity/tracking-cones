import bpy
import math
import os

# --- GPU CONFIGURATION ---
def enable_gpus():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.film_transparent = True
    
    # Access Workbench-specific shading settings
    shading = scene.display.shading

    shading.light = 'FLAT'             # Removes 3D shadows
    shading.color_type = 'OBJECT'      # Uses the color set in obj.color
    shading.show_object_outline = False # Turns on the silhouette 
    
    # Setting the background to black
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (0, 0, 0, 1)

# --- CONSTANTS ---
DISTANCE = 400
POSES = []
STL_PATH = "/data/20mm Tracking Cone.stl"
OUTPUT_DIR = "/data/renders_400"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_poses():
    for angle in range(0, 68, 1):
        y = -DISTANCE * math.cos(math.radians(angle))
        z = DISTANCE * math.sin(math.radians(angle))
        rot_rad = (math.radians(90 - angle), 0, 0)
        POSES.append({"loc": (0.0, y, z), "rot": rot_rad, "name": f"render_{angle:02d}_{DISTANCE}"})

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_scene():
    # Import STL
    bpy.ops.wm.stl_import(filepath=STL_PATH)
    obj = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 10)
    
    # Make the object black so only the outline shows
    obj.color = (1, 1, 1, 1)

    return obj

def create_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.clip_end = 2020
    bpy.context.scene.camera = cam
    # Ensure camera is focused on origin
    return cam

def render_poses(cam):
    for pose in POSES:
        cam.location = pose["loc"]
        cam.rotation_euler = pose["rot"]

        asin = math.sin(pose["rot"][0])
        acos = math.cos(pose["rot"][0])
        bsin = math.sin(pose["rot"][1])
        bcos = math.cos(pose["rot"][1])
        csin = math.sin(pose["rot"][2])
        ccos = math.cos(pose["rot"][2])

        name = f"{pose['name']}_{pose['loc'][0]:.2f}_{pose['loc'][1]:.2f}_{pose['loc'][2]:.2f}_{asin:.2f}_{acos:.2f}_{bsin:.2f}_{bcos:.2f}_{csin:.2f}_{ccos:.2f}.png"

        file_path = os.path.join(OUTPUT_DIR, name)
        bpy.context.scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)
        print(f"Saved: {file_path}")

# Execute
print("Starting Outline Render...")
clear_scene()
enable_gpus() 
generate_poses()
obj = setup_scene()
cam = create_camera()
render_poses(cam)
print("Done!")
