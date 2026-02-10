import numpy as np
import matplotlib.pyplot as plt

AREA_DATA_PATH = "/data/area_data.csv"
pixel_count = 2000
spread = 100

# Load the data.
data = np.loadtxt(AREA_DATA_PATH, delimiter=",", skiprows=1)
data_distance = data[:, 0]
data_angle = data[:, 1]
data_area = data[:, 2]
unique_distances = np.unique(data_distance)

# Create a figure and axis.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot ax.
for dist in reversed(unique_distances):
    # Filter the data for the current distance series
    mask = (data_distance == dist) & (data_area >= pixel_count - spread) & (data_area <= pixel_count + spread)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask]
    sort_idx = np.argsort(x_values)

    if y_values.size == 0:
        continue

    # Plot the specific series
    ax.plot(x_values[sort_idx], dist * np.ones_like(x_values[sort_idx]), y_values[sort_idx],
             label=f"Distance: {dist}", 
             marker="o",
             markersize=1)

# Set the title and labels.
ax.set_title(f"Pixel Count by Angle and Distance @ {pixel_count} (Spread: {spread})")
ax.set_xlabel("Angle")
ax.set_ylabel("Distance")
ax.set_zlabel("Pixel Count")

for azim in range(0, 360, 1):
    ax.view_init(elev=20, azim=azim)
    plt.savefig(f"/data/area-pixel-count-spread-slice/{pixel_count}-{spread}-{azim:03d}.png")
