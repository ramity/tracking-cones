from PIL import Image
import glob

# Find all PNG files in the current directory, sorted by name
# You can change the path and extension as needed
file_list = sorted(glob.glob('/data/contours_120/*.png')) 

# Create a list to store the image objects
images = []

for filename in file_list:
    img = Image.open(filename)
    images.append(img)

# Save the first image, and append the rest as frames
if images:
    images[0].save(
        '/data/output.gif',
        format='GIF',
        append_images=images[1:],
        save_all=True,
        duration=50,
        loop=0,
        disposal=2
    )
    print(f"Successfully created output.gif from {len(images)} images.")
else:
    print("No PNG images found to convert.")
