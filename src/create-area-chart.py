import numpy as np
import matplotlib.pyplot as plt

AREA_DATA_PATH = "/data/area_data.csv"
spread = 500

# Load the data.
data = np.loadtxt(AREA_DATA_PATH, delimiter=",", skiprows=1)
data_distance = data[:, 0]
data_angle = data[:, 1]
data_area = data[:, 2]
unique_distances = np.unique(data_distance)

# # Create the figure and axes.
# fig, ax = plt.subplots(1, 1, figsize=(20, 20))

# # Calculate the best fit line for distance vs pixels.
# x_values = []
# y_values = []
# for dist in unique_distances:
#     # Filter the data for the current distance series
#     mask = (data_distance == dist) & (data_angle == 67)
    
#     # Extract and sort values by angle to ensure lines connect properly
#     x_values.append(data_distance[mask][0])
#     y_values.append(data_area[mask][0])

# coefficients = np.polyfit(x_values, y_values, 2)
# polynomial = np.poly1d(coefficients)

# x_values = np.linspace(105, 1005, 100) 

# # Plot the best fit line.
# ax.plot(x_values, polynomial(x_values), color="red", linewidth=2)
# plt.savefig("/data/area_chart.png")

# import sys
# sys.exit()

# Create the figure and axes.
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, 1, figsize=(20, 40))

fig.tight_layout(pad=5.0)

# Plot ax1.
for dist in unique_distances:
    # Filter the data for the current distance series
    mask = (data_distance == dist)
    
    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask]
    sort_idx = np.argsort(x_values)
    
    # Plot the specific series
    ax1.plot(x_values[sort_idx], y_values[sort_idx], 
             label=f"Distance: {dist}", 
             marker="o", 
             markersize=4, 
             linestyle="-")

# Plot ax2.
for dist in unique_distances:
    # Filter the data for the current distance series
    mask = (data_distance == dist)
    
    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask] / dist**2
    sort_idx = np.argsort(x_values)
    
    # Plot the specific series
    ax2.plot(x_values[sort_idx], y_values[sort_idx], 
             label=f"Distance: {dist}", 
             marker="o", 
             markersize=4, 
             linestyle="-")

# Plot ax3.
for dist in unique_distances:
    # Filter the data for the current distance series
    mask = (data_distance == dist) & (data_area >= 80000 - spread) & (data_area <= 80000 + spread)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask]
    sort_idx = np.argsort(x_values)

    if y_values.size == 0:
        continue

    # Plot the specific series
    ax3.plot(x_values[sort_idx], y_values[sort_idx], 
             label=f"Distance: {dist}", 
             marker="o", 
             markersize=4, 
             linestyle="-")

# Plot ax4.
ax4_x_values = []
ax4_y_values = []
for dist in unique_distances:
    # Filter the data for the current distance series
    mask = (data_distance == dist) & (data_area >= 80000 - spread) & (data_area <= 80000 + spread)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask] / dist**2
    sort_idx = np.argsort(x_values)

    if y_values.size == 0:
        continue

    # Plot the specific series
    ax4.plot(x_values[sort_idx], y_values[sort_idx], 
             label=f"Distance: {dist}", 
             marker="o", 
             markersize=4, 
             linestyle="-")

    ax4_x_values.extend(x_values)
    ax4_y_values.extend(y_values)

# Calculate the best fit line for ax4.
ax4_coefficients = np.polyfit(ax4_x_values, ax4_y_values, 3)
ax4_polynomial = np.poly1d(ax4_coefficients)
print(ax4_coefficients)

# Plot the best fit line.
ax4.plot(ax4_x_values, ax4_polynomial(ax4_x_values), color="red", linewidth=2)

# Plot ax5.
for dist in unique_distances:
    # Filter the data for the current distance series
    mask = (data_distance == dist) & (data_area >= 60000 - spread) & (data_area <= 60000 + spread)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask]
    sort_idx = np.argsort(x_values)

    if y_values.size == 0:
        continue

    # Plot the specific series
    ax5.plot(x_values[sort_idx], y_values[sort_idx], 
             label=f"Distance: {dist}", 
             marker="o", 
             markersize=4, 
             linestyle="-")

# Plot ax6.
ax6_x_values = []
ax6_y_values = []
for dist in unique_distances:
    # Filter the data for the current distance series
    mask = (data_distance == dist) & (data_area >= 60000 - spread) & (data_area <= 60000 + spread)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask] / dist**2
    sort_idx = np.argsort(x_values)

    if y_values.size == 0:
        continue

    ax6_x_values.extend(x_values)
    ax6_y_values.extend(y_values)

    # Plot the specific series
    ax6.plot(x_values[sort_idx], y_values[sort_idx], 
             label=f"Distance: {dist}", 
             marker="o", 
             markersize=4, 
             linestyle="-")

# Calculate the best fit line for ax6.
ax6_coefficients = np.polyfit(ax6_x_values, ax6_y_values, 3)
ax6_polynomial = np.poly1d(ax6_coefficients)
print(ax6_coefficients)

# Plot the best fit line.
ax6.plot(ax6_x_values, ax6_polynomial(ax6_x_values), color="red", linewidth=2)

# Plot ax7.
for dist in unique_distances:
    # Filter the data for the current distance series
    mask = (data_distance == dist) & (data_area >= 70000 - spread) & (data_area <= 70000 + spread)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask]
    sort_idx = np.argsort(x_values)

    if y_values.size == 0:
        continue

    # Plot the specific series
    ax7.plot(x_values[sort_idx], y_values[sort_idx], 
             label=f"Distance: {dist}", 
             marker="o", 
             markersize=4, 
             linestyle="-")

# Plot ax8.
ax8_x_values = []
ax8_y_values = []
for dist in unique_distances:
    # Filter the data for the current distance series
    mask = (data_distance == dist) & (data_area >= 70000 - spread) & (data_area <= 70000 + spread)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask] / dist**2
    sort_idx = np.argsort(x_values)

    if y_values.size == 0:
        continue

    # Plot the specific series
    ax8.plot(x_values[sort_idx], y_values[sort_idx], 
             label=f"Distance: {dist}", 
             marker="o", 
             markersize=4, 
             linestyle="-")

    ax8_x_values.extend(x_values)
    ax8_y_values.extend(y_values)

# Calculate the best fit line for ax4.
ax8_coefficients = np.polyfit(ax8_x_values, ax8_y_values, 3)
ax8_polynomial = np.poly1d(ax8_coefficients)
print(ax8_coefficients)

# Plot ax4, ax6, and ax8 trendlines on ax9.
ax9.plot(ax4_x_values, ax4_polynomial(ax4_x_values), label=f"Area = 80000 +/- {spread}", color="red", linewidth=2)
ax9.plot(ax6_x_values, ax6_polynomial(ax6_x_values), label=f"Area = 60000 +/- {spread}", color="blue", linewidth=2)
ax9.plot(ax8_x_values, ax8_polynomial(ax8_x_values), label=f"Area = 70000 +/- {spread}", color="green", linewidth=2)

# Plot ax10.
distance_polynomials = []
for dist in unique_distances:
    # Filter the data for the current distance series
    mask = (data_distance == dist)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask]

    # Calculate the best fit line for ax10.
    coefficients = np.polyfit(x_values, y_values, 3)
    polynomial = np.poly1d(coefficients)
    distance_polynomials.append(polynomial)

    # Plot the specific series
    ax10.plot(x_values, polynomial(x_values), label=f"Distance: {dist}", marker="o", markersize=4, linestyle="-")

# Set the titles, labels, and legends.
ax1.set_title("Relationship of Area vs Angle grouped by Distance", fontsize=14)
ax1.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax1.set_ylabel("Pixels", fontsize=12)
ax1.legend(title="Distance Series")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.set_title("Relationship of Area/Distance^2 vs Angle grouped by Distance", fontsize=14)
ax2.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax2.set_ylabel("Pixels / Distance^2", fontsize=12)
ax2.legend(title="Distance Series")
ax2.grid(True, linestyle="--", alpha=0.6)

ax3.set_title(f"Relationship of Area vs Angle grouped by Distance (Where Area = 80000 +/- {spread})", fontsize=14)
ax3.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax3.set_ylabel("Pixels", fontsize=12)
ax3.legend(title="Distance Series")
ax3.grid(True, linestyle="--", alpha=0.6)

ax4.set_title(f"Relationship of Area/Distance^2 vs Angle grouped by Distance (Where Area = 80000 +/- {spread})", fontsize=14)
ax4.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax4.set_ylabel("Pixels / Distance^2", fontsize=12)
ax4.legend(title="Distance Series")
ax4.grid(True, linestyle="--", alpha=0.6)

ax5.set_title(f"Relationship of Area vs Angle grouped by Distance (Where Area = 60000 +/- {spread})", fontsize=14)
ax5.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax5.set_ylabel("Pixels", fontsize=12)
ax5.legend(title="Distance Series")
ax5.grid(True, linestyle="--", alpha=0.6)

ax6.set_title(f"Relationship of Area/Distance^2 vs Angle grouped by Distance (Where Area = 60000 +/- {spread})", fontsize=14)
ax6.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax6.set_ylabel("Pixels / Distance^2", fontsize=12)
ax6.legend(title="Distance Series")
ax6.grid(True, linestyle="--", alpha=0.6)

ax7.set_title(f"Relationship of Area vs Angle grouped by Distance (Where Area = 70000 +/- {spread})", fontsize=14)
ax7.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax7.set_ylabel("Pixels", fontsize=12)
ax7.legend(title="Distance Series")
ax7.grid(True, linestyle="--", alpha=0.6)

ax8.set_title(f"Relationship of Area/Distance^2 vs Angle grouped by Distance (Where Area = 70000 +/- {spread})", fontsize=14)
ax8.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax8.set_ylabel("Pixels / Distance^2", fontsize=12)
ax8.legend(title="Distance Series")
ax8.grid(True, linestyle="--", alpha=0.6)

ax9.set_title("Relationship of Area/Distance^2 vs Angle grouped by Distance (Where Area = 80000 +/- {spread} and Area/Distance^2 = 60000 +/- {spread})", fontsize=14)
ax9.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax9.set_ylabel("Pixels / Distance^2", fontsize=12)
ax9.legend(title="Distance Series")
ax9.grid(True, linestyle="--", alpha=0.6)

ax10.set_title("LoBF curves for Pixels vs Angle", fontsize=14)
ax10.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax10.set_ylabel("Pixels", fontsize=12)
ax10.legend(title="Distance Series")
ax10.grid(True, linestyle="--", alpha=0.6)

# Save the figure.
plt.savefig("/data/line_chart_by_distance.png")

## Revisiting cropping/clipping complete graph by limiting and clip_on

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the 3D surface @ 60000 pixels.
# x = np.linspace(0, 90, 100)
# y = np.linspace(105, 205, 100)
# x, y = np.meshgrid(x, y)
# Z = np.linspace(60000, 60000, 100).reshape(1, -1)
# Z = np.repeat(Z, 100, axis=0)
# ax.plot_surface(x, y, Z, alpha=1, color="black")

for dist in reversed(unique_distances):
    # Filter the data for the current distance series
    mask = (data_distance == dist) & (data_area >= 60000) & (data_area <= 80000)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask]

    if x_values.size == 0 or y_values.size == 0:
        continue

    # Plot the specific series
    ax.scatter(x_values, dist * np.ones(len(x_values)), y_values, label=f"Distance: {dist}", marker="o", s=2)

ax.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax.set_ylabel("Distance (mm)", fontsize=12)
ax.set_zlabel("Pixels", fontsize=12)
ax.grid(True, linestyle="-", alpha=1)
ax.set_xlim([0, 67])
ax.set_ylim([135, 188])
ax.set_zlim([60000, 80000])

ax.view_init(elev=15, azim=-45)
plt.savefig("/data/3d_surface_by_distance_slice_limited.png")

ax.view_init(elev=0, azim=0)
plt.savefig("/data/3d_surface_by_distance_slice_limited_0.png")

ax.view_init(elev=0, azim=-90)
plt.savefig("/data/3d_surface_by_distance_slice_limited_-90.png")

# Rotate the camera around the surface and save images.
frame = 0
for i in range(-1, -90, -1):
    ax.view_init(elev=30, azim=i)
    plt.savefig(f"/data/3d_surface_by_distance_slice_limited/{frame:03d}.png")
    frame += 1
for i in range(-89, 0, 1):
    ax.view_init(elev=30, azim=i)
    plt.savefig(f"/data/3d_surface_by_distance_slice_limited/{frame:03d}.png")
    frame += 1

# Plot the 3D surface created by the LoBF curves.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for dist in reversed(unique_distances):
    # Filter the data for the current distance series
    mask = (data_distance == dist) & (data_area >= 80000 - 2500) & (data_area <= 80000 + 2500)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask]

    if x_values.size == 0 or y_values.size == 0:
        continue

    # Plot the specific series
    ax.plot(x_values, dist * np.ones(len(x_values)), y_values, label=f"Distance: {dist}", marker="o", markersize=1, linestyle="-")

ax.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax.set_ylabel("Distance (mm)", fontsize=12)
ax.set_zlabel("Pixels", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)
plt.savefig("/data/3d_surface_by_distance_slice.png")
ax.view_init(elev=0, azim=-90)
plt.savefig("/data/3d_surface_by_distance_slice_-90.png")
ax.view_init(elev=0, azim=0)
plt.savefig("/data/3d_surface_by_distance_slice_0.png")

# Rotate the camera around the surface and save images.
frame = 0
for i in range(-1, -90, -1):
    ax.view_init(elev=30, azim=i)
    plt.savefig(f"/data/3d_surface_animation_slice/{frame:03d}.png")
    frame += 1
for i in range(-89, 0, 1):
    ax.view_init(elev=30, azim=i)
    plt.savefig(f"/data/3d_surface_animation_slice/{frame:03d}.png")
    frame += 1

# Plot the 3D surface created by the LoBF curves.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for dist in reversed(unique_distances):
    # Filter the data for the current distance series
    mask = (data_distance == dist)

    # Extract and sort values by angle to ensure lines connect properly
    x_values = data_angle[mask]
    y_values = data_area[mask]

    # Calculate the best fit line for ax10.
    coefficients = np.polyfit(x_values, y_values, 3)
    polynomial = np.poly1d(coefficients)
    distance_polynomials.append(polynomial)

    # Plot the specific series
    ax.plot(x_values, dist * np.ones(len(x_values)), polynomial(x_values), label=f"Distance: {dist}", marker="o", markersize=1, linestyle="-")

ax.set_xlabel("Angle (theta in degrees)", fontsize=12)
ax.set_ylabel("Distance (mm)", fontsize=12)
ax.set_zlabel("Pixels", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)
plt.savefig("/data/3d_surface_by_distance.png")

# Rotate the camera around the surface and save images.
frame = 0
for i in range(-1, -90, -1):
    ax.view_init(elev=30, azim=i)
    plt.savefig(f"/data/3d_surface_animation/{frame:03d}.png")
    frame += 1
for i in range(-89, 0, 1):
    ax.view_init(elev=30, azim=i)
    plt.savefig(f"/data/3d_surface_animation/{frame:03d}.png")
    frame += 1
