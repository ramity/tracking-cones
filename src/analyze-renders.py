import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import math

known_cone_width = 20
known_cone_height = 20
known_cone_slant_height = math.sqrt(known_cone_width**2 + known_cone_height**2)
known_cone_radius = known_cone_width / 2

focal_length = 50
sensor_width = 36
sensor_height = 20
pixel_spread = 250
angle_spread = 10

AREA_DATA_PATH = "/data/area_data.csv"
render_directory = "/data/renders/"
output_csv_path = "/data/render_analysis.csv"

renders = os.listdir(render_directory)

# renders = renders[47627:47671]

# with open(output_csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow([
#         'render_id', 
#         'known_angle', 
#         'inclination_angle', 
#         'tilt_angle', 
#         'pixel_queried_angle', 
#         'known_distance',
#         'pnp_distance', 
#         'base_pixel_distance',
#         'queried_distance', 
#         'major_axis', 
#         'minor_axis', 
#         'query_pixel_count', 
#         'query_pixel_spread', 
#         'query_angle_spread', 
#         'query_results', 
#         'closest_pixel_count'
#     ])

renders = renders[35:61235:68]

# Fix to handle :03d vs :04d in filenames.
# Sort renders by render_DISTANCE_ANGLE.png.
renders.sort(key=lambda x: int(x.split('_')[1]))

for render_id, render in enumerate(renders):

    # Skip non-png files.
    if not render.endswith(".png"):
        continue

    render_path = os.path.join(render_directory, render)
    image = cv2.imread(render_path)
    image_height, image_width, _ = image.shape

    # Extract distance and angle from filename.
    known_distance = render.split('_')[1]
    known_angle = render.split('_')[2].split('.')[0]

    # Convert to grayscale, threshold, and count white pixels.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    white_pixels = np.sum(thresh == 255)

    # Find the contours of thresholded image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours.
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    # cv2.imwrite("/data/test/" + render, image)

    # Definitions:
    # - Top is the apex of the cone.
    # - Bottom is the closest point to the camera on the base of the cone.
    # - Left is the left side of the cone.
    # - Right is the right side of the cone.
    # - Center is the center of the base of the cone.

    # Determine the top point.
    top_y = min([c[0][1] for c in contours[0]])
    top_most_contours = [c for c in contours[0] if c[0][1] == top_y]
    top_center_x = np.mean([c[0][0] for c in top_most_contours])
    top = (top_center_x, top_y)
    top_obj = (0, 0, 20)

    # Determine the bottom point.
    bottom_y = max([c[0][1] for c in contours[0]])
    bottom_most_contours = [c for c in contours[0] if c[0][1] == bottom_y]
    bottom_center_x = np.mean([c[0][0] for c in bottom_most_contours])
    bottom = (bottom_center_x, bottom_y)
    bottom_obj = (0, 10, 0)

    # Determine the left point.
    left_x = min([c[0][0] for c in contours[0]])
    left_most_contours = [c for c in contours[0] if c[0][0] == left_x]
    left_center_y = np.mean([c[0][1] for c in left_most_contours])
    left = (left_x, left_center_y)
    left_obj = (10, 0, 0)

    # Determine the right point.
    right_x = max([c[0][0] for c in contours[0]])
    right_most_contours = [c for c in contours[0] if c[0][0] == right_x]
    right_center_y = np.mean([c[0][1] for c in right_most_contours])
    right = (right_x, right_center_y)
    right_obj = (-10, 0, 0)

    # Determine the center point.
    center_x = (left[0] + right[0]) / 2
    center_y = (left[1] + right[1]) / 2
    center = (center_x, center_y)
    center_obj = (0, 0, 0)

    # Determine the back point using bottom and center points.
    back_x = center[0] + (center[0] - bottom[0])
    back_y = center[1] + (center[1] - bottom[1])
    back = (back_x, back_y)
    back_obj = (0, -10, 0)

    # Determine the base width pixel distance.
    base_width_pixel_distance = math.sqrt((left[0] - right[0])**2 + (left[1] - right[1])**2)

    # Determine the cone height pixel distance.
    cone_height_pixel_distance = math.sqrt((top[0] - center[0])**2 + (top[1] - center[1])**2)

    # Determine the slant height pixel distance.
    slant_height_pixel_distance = math.sqrt((top[0] - bottom[0])**2 + (top[1] - bottom[1])**2)

    # Determine the major and minor axis of the ellipse.
    left_to_center_distance = math.sqrt((left[0] - center[0])**2 + (left[1] - center[1])**2)
    right_to_center_distance = math.sqrt((right[0] - center[0])**2 + (right[1] - center[1])**2)
    center_to_bottom_distance = math.sqrt((center[0] - bottom[0])**2 + (center[1] - bottom[1])**2)
    major_axis = max(left_to_center_distance, right_to_center_distance)
    minor_axis = center_to_bottom_distance

    # Draw the top, bottom, left, right, and center points.
    cv2.ellipse(output, (int(center[0]), int(center[1])), (int(major_axis), int(minor_axis)), 0, 0, 360, (255, 255, 255), 1)
    cv2.circle(output, (int(top[0]), int(top[1])), 0, (0, 0, 255), -1)
    cv2.circle(output, (int(bottom[0]), int(bottom[1])), 0, (0, 0, 255), -1)
    cv2.circle(output, (int(left[0]), int(left[1])), 0, (255, 0, 0), -1)
    cv2.circle(output, (int(right[0]), int(right[1])), 0, (255, 255, 0), -1)
    cv2.circle(output, (int(center[0]), int(center[1])), 0, (0, 255, 255), -1)
    cv2.circle(output, (int(back[0]), int(back[1])), 0, (255, 0, 255), -1)

    # Determine the inclination angle of the cone.
    inclination_angle = math.degrees(math.asin(minor_axis / major_axis))

    # Estimate the distance of the cone from the camera.
    pixel_pitch_width = sensor_width / image_width
    pixel_pitch_height = sensor_height / image_height
    distance = (known_cone_width * focal_length * image_width) / (base_width_pixel_distance * sensor_width)

    obj_points = np.array([top_obj, bottom_obj, left_obj, right_obj, center_obj, back_obj], dtype=np.float32)
    img_points = np.array([top, bottom, left, right, center, back], dtype=np.float32)
    camera_matrix = np.array([[focal_length * image_width / sensor_width, 0, image_width / 2], [0, focal_length * image_height / sensor_height, image_height / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    pnp_distance = np.linalg.norm(tvec)
    R, _ = cv2.Rodrigues(rvec)
    camera_pos = -np.matrix(R).T @ np.matrix(tvec)
    # print("camera_pos: " + str(camera_pos))

    # The tilt of the camera relative to the cone's vertical axis
    # Extract the dot product of the camera's Z-axis and the cone's vertical axis
    R, _ = cv2.Rodrigues(rvec)
    tilt_angle = np.degrees(np.arccos(R[2, 2])) - 90

    data = np.loadtxt(AREA_DATA_PATH, delimiter=",", skiprows=1)
    data_distance = data[:, 0]
    data_angle = data[:, 1]
    data_area = data[:, 2]

    average_angle = np.mean([inclination_angle, tilt_angle])
    mask = (data_area >= white_pixels - pixel_spread) & (data_area <= white_pixels + pixel_spread) & (data_angle >= average_angle - angle_spread) & (data_angle <= average_angle + angle_spread)
    results = data[mask]
    results = results[results[:, 2].argsort()]
    min_distance = results[np.argmin(results[:, 0])][0]
    max_distance = results[np.argmax(results[:, 0])][0]
    min_angle = results[np.argmin(results[:, 1])][1]
    max_angle = results[np.argmax(results[:, 1])][1]

    # Find the result with the smallest difference in both angle and pixel count.
    diff = np.sqrt((results[:, 1] - average_angle)**2 + (results[:, 2] - white_pixels)**2)
    closest_idx = np.argmin(diff)
    closest_distance = results[closest_idx, 0]
    closest_angle = results[closest_idx, 1]
    closest_pixel_count = results[closest_idx, 2]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.tight_layout()

    # Set ax1 to be the image.
    ax1.imshow(output)
    ax1.set_title("Input Image")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True)

    ax2.autoscale(enable=False, axis='x')
    ax2.autoscale(enable=False, axis='y')
    min_distance = int(known_distance) - int(int(known_distance) * 0.05)
    max_distance = int(known_distance) + int(int(known_distance) * 0.05)
    min_angle = int(known_angle) - angle_spread
    max_angle = int(known_angle) + angle_spread

    scatter = ax2.scatter(results[:, 0], results[:, 1], c=results[:, 2], label='pixel_count', cmap='coolwarm')
    fig.colorbar(scatter, ax=ax2, location='right', anchor=(0, 0), shrink=0.7)
    ax2.plot([min_distance, max_distance], [int(known_angle), int(known_angle)], label='known_angle', linestyle='-', color='red')
    ax2.plot([int(known_distance), int(known_distance)], [min_angle, max_angle], label='known_distance', linestyle='-', color='red')
    ax2.plot([min_distance, max_distance], [inclination_angle, inclination_angle], label='inclination_angle', linestyle='-', color='orange', linewidth=1)
    ax2.plot([distance, distance], [min_angle, max_angle], label='pixel_distance', linestyle='-', color='orange', linewidth=1)
    ax2.plot([min_distance, max_distance], [tilt_angle, tilt_angle], label='tilt_angle', linestyle='-', color='blue', linewidth=1)
    ax2.plot([pnp_distance, pnp_distance], [min_angle, max_angle], label='pnp_distance', linestyle='-', color='blue', linewidth=1)
    ax2.plot([closest_distance, closest_distance], [min_angle, max_angle], label='queried_distance', linestyle='-', color='green')
    ax2.plot([min_distance, max_distance], [closest_angle, closest_angle], label='queried_angle', linestyle='-', color='green')
    ax2.set_xlabel("Distance (mm)")
    ax2.set_ylabel("Angle (degrees)")
    ax2.set_xlim([min_distance, max_distance])
    ax2.set_ylim([min_angle, max_angle])
    ax2.set_title(f"Pixel Count {white_pixels} pixel_spread {pixel_spread} angle_spread {angle_spread} bounds Distance")
    ax2.legend(loc='lower right')
    ax2.grid(True)

    # Save plot.
    plt.savefig(f"/data/query-results/{render}")
    # plt.savefig("/data/test.png")

    plt.close(fig)

    # import sys
    # sys.exit(0)

    # Debug prints:
    # print("known angle:\t\t\t" + str(int(known_angle)))
    # print("- pnp estimated angle:\t\t" + str(tilt_angle))
    # print("- base pixel estimated angle:\t" + str(inclination_angle))
    # print("- averaged angle:\t\t" + str(average_angle))
    # # print("pnp success:\t\t\t" + str(success))
    # # print("rvec: " + str(rvec))
    # # print("tvec: " + str(tvec))
    # print("known distance:\t\t\t" + str(known_distance))
    # print("- pnp distance:\t\t\t" + str(pnp_distance))
    # print("- pixel distance:\t\t" + str(distance))
    # print("- queried distance:\t\t" + str(closest_distance))
    # print("query pixel count:\t\t" + str(white_pixels))
    # print("query pixel spread:\t\t" + str(pixel_spread))
    # print("query angle spread:\t\t" + str(angle_spread))
    # print("query results:\t\t\t" + str(len(results)))
    # print("closest angle:\t\t\t" + str(closest_angle))
    # print("closest pixel count:\t\t" + str(closest_pixel_count))

    # Write row to CSV file.

    # row = [
    #     render_id,
    #     known_angle,
    #     inclination_angle,
    #     tilt_angle,
    #     closest_angle,
    #     known_distance,
    #     pnp_distance,
    #     distance,
    #     closest_distance,
    #     major_axis,
    #     minor_axis,
    #     white_pixels,
    #     pixel_spread,
    #     angle_spread,
    #     len(results),
    #     closest_pixel_count
    # ]
    # writer.writerow(row)
