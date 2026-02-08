import bpy
import os
import math

# Clear existing scene (optional - removes default cube if you want to start fresh)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Ensure we have the default cube
# If you deleted it above, uncomment these lines to recreate it:
# bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Get the default cube (assuming it exists)
bpy.ops.wm.stl_import(filepath="/data/20mm Tracking Cone.stl")
obj = bpy.context.selected_objects[0]
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.location = (0, 0, 10)

# Make the object black so only the outline shows
# obj.color = (0, 0, 1, 1)

# if obj:
    # obj.select_set(True)
    # bpy.context.view_layer.objects.active = obj

# Set up camera
bpy.ops.object.camera_add()
camera = bpy.context.active_object
bpy.context.scene.camera = camera
camera.location = (0, -100, 0)
camera.rotation_euler = (math.radians(90), 0, 0)

# Enable Freestyle for outline rendering
# bpy.context.scene.render.use_freestyle = True

# Configure Freestyle settings for clean outlines
freestyle_settings = bpy.context.scene.view_layers[0].freestyle_settings
freestyle_settings.mode = 'EDITOR'

# Create a new lineset for the outline
lineset = freestyle_settings.linesets.new("Outline")
# lineset.select_silhouette = True
# lineset.select_border = True
# lineset.select_crease = True

# Configure line style
linestyle = lineset.linestyle
linestyle.color = (255, 0, 255)
linestyle.thickness = 5.0

# Set render settings
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.film_transparent = True

# Set output path
output_path = "/data/cube_outline.png"
bpy.context.scene.render.filepath = output_path
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Render
print(f"Rendering cube outline to {output_path}...")
bpy.ops.render.render(animation=False, write_still=True)
print(f"Render complete! Saved to {output_path}")