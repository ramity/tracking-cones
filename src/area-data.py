import bpy
import math
import os

STL_PATH = "/data/20mm Tracking Cone.stl"
OUTPUT_PATH = "/data/area_data.csv"
FOCAL_LENGTH = 50
KNOWN_WIDTH = 20
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
SENSOR_WIDTH = 36
SENSOR_HEIGHT = 24
ANGLES = range(0, 68, 1)
DISTANCES = range(105, 1006, 1)

# Ensure clean slate
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Import STL
bpy.ops.wm.stl_import(filepath=STL_PATH)
obj = bpy.context.selected_objects[0]
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
obj.location = (0, 0, 10)

# Make the object black so only the outline shows
obj.color = (1, 1, 1, 1)

# Create camera
bpy.ops.object.camera_add()
cam = bpy.context.active_object
bpy.context.scene.camera = cam

# Configure scene
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.film_transparent = True
shading = scene.display.shading
shading.light = "FLAT"
shading.color_type = "OBJECT"

# For each distance:
for distance in DISTANCES:

    # Set the camera"s clip end to be slightly larger than the distance.
    cam.data.clip_end = distance + 20

    # For each angle:
    for angle in ANGLES:

        # Check if file already exists.
        filename = f"render_{distance:04d}_{angle:03d}.png"
        filepath = f"/data/renders/{filename}"
        if os.path.exists(filepath):
            continue

        # Calculate the camera"s position given the distance and angle.
        y = -distance * math.cos(math.radians(angle))
        z = distance * math.sin(math.radians(angle))
        cam.location = (0, y, z)
        euler = (math.radians(90 - angle), 0, 0)
        cam.rotation_euler = euler

        # Render the image.
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
