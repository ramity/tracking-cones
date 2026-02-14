import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import math
import sys

known_cone_width = 20
known_cone_height = 20
known_cone_slant_height = math.sqrt(known_cone_width**2 + known_cone_height**2)
known_cone_radius = known_cone_width / 2

focal_length = 5.4 # validate
sensor_width = 8.16 # validate
sensor_height = 6.14 # validate
pixel_spread = 250
angle_spread = 10

render_width = 1920
render_height = 1080

AREA_DATA_PATH = "/data/area_data.csv"
render_directory = "/data/irl/"

renders = os.listdir(render_directory)

# output_csv_path = "/data/render_analysis.csv"
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

# Fix to handle :03d vs :04d in filenames.
# Sort renders by render_DISTANCE_ANGLE.png.
# renders.sort(key=lambda x: int(x.split('_')[1]))

contour_selections = [
    48666,
    45495,
    35385,
    33115,
    14412,
    19564,
    30966,
    39622,
]

for render_id, (render, contour_selection) in enumerate(zip(renders, contour_selections)):

    render_path = os.path.join(render_directory, render)
    image = cv2.imread(render_path)
    image_height, image_width, _ = image.shape
    convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale, threshold, and count white pixels.
    output = np.zeros_like(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 135, 255, 0)

    # Find the contours of thresholded image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours in order of area.
    contours = list(contours)
    contours.sort(key=cv2.contourArea)

    if contour_selection == -1:
        for i, contour in enumerate(contours):
            cv2.putText(output, str(i), (int(contour[0][0][0]), int(contour[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.drawContours(output, contour, -1, (255, 255, 255), 1)
        cv2.imwrite("/data/test.png", output)
        sys.exit()

    selected_contour = contours[contour_selection]
    cv2.drawContours(output, [selected_contour], -1, (255, 255, 255), -1)

    # Definitions:
    # - Top is the apex of the cone.
    # - Bottom is the closest point to the camera on the base of the cone.
    # - Left is the left side of the cone.
    # - Right is the right side of the cone.
    # - Center is the center of the base of the cone.

    # Determine the top point.
    top_y = min([c[0][1] for c in selected_contour])
    top_most_contours = [c for c in selected_contour if c[0][1] == top_y]
    top_center_x = np.mean([c[0][0] for c in top_most_contours])
    top = (top_center_x, top_y)
    top_obj = (0, 0, 20)

    # Determine the bottom point.
    bottom_y = max([c[0][1] for c in selected_contour])
    bottom_most_contours = [c for c in selected_contour if c[0][1] == bottom_y]
    bottom_center_x = np.mean([c[0][0] for c in bottom_most_contours])
    bottom = (bottom_center_x, bottom_y)
    bottom_obj = (0, 10, 0)

    # Determine the left point.
    left_x = min([c[0][0] for c in selected_contour])
    left_most_contours = [c for c in selected_contour if c[0][0] == left_x]
    left_center_y = np.mean([c[0][1] for c in left_most_contours])
    left = (left_x, left_center_y)
    left_obj = (10, 0, 0)

    # Determine the right point.
    right_x = max([c[0][0] for c in selected_contour])
    right_most_contours = [c for c in selected_contour if c[0][0] == right_x]
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
    white_pixels = np.sum(output == 255)

    cv2.circle(output, (int(top[0]), int(top[1])), 1, (0, 0, 255), -1)
    cv2.circle(output, (int(bottom[0]), int(bottom[1])), 1, (0, 0, 255), -1)
    cv2.circle(output, (int(left[0]), int(left[1])), 1, (255, 0, 0), -1)
    cv2.circle(output, (int(right[0]), int(right[1])), 1, (255, 255, 0), -1)
    cv2.circle(output, (int(center[0]), int(center[1])), 1, (0, 255, 255), -1)
    cv2.circle(output, (int(back[0]), int(back[1])), 1, (255, 0, 255), -1)
    cv2.imwrite("/data/test.png", output)

    # Determine the inclination angle of the cone.
    inclination_angle = math.degrees(math.asin(minor_axis / major_axis))

    # Estimate the distance of the cone from the camera.
    distance = (known_cone_width * focal_length * image_width) / (base_width_pixel_distance * sensor_width)

    obj_points = np.array([top_obj, bottom_obj, left_obj, right_obj, center_obj, back_obj], dtype=np.float32)
    img_points = np.array([top, bottom, left, right, center, back], dtype=np.float32)
    camera_matrix = np.array([[focal_length * image_width / sensor_width, 0, image_width / 2], [0, focal_length * image_height / sensor_height, image_height / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    pnp_distance = np.linalg.norm(tvec)
    R, _ = cv2.Rodrigues(rvec)
    camera_pos = -np.matrix(R).T @ np.matrix(tvec)

    # The tilt of the camera relative to the cone's vertical axis
    # Extract the dot product of the camera's Z-axis and the cone's vertical axis
    R, _ = cv2.Rodrigues(rvec)
    tilt_angle = np.degrees(np.arccos(R[2, 2])) - 90

    data = np.loadtxt(AREA_DATA_PATH, delimiter=",", skiprows=1)
    data_distance = data[:, 0]
    data_angle = data[:, 1]
    data_area = data[:, 2]

    # Scale the pixel count to consider the image_width, render_width and image_height, render_height deltas.
    scale_x = render_width / image_width
    scale_y = render_height / image_height
    white_pixels = white_pixels * scale_x * scale_y

    print(white_pixels)

    average_angle = np.mean([inclination_angle, tilt_angle])
    mask = (data_area >= white_pixels - pixel_spread) & (data_area <= white_pixels + pixel_spread) & (data_angle >= average_angle - angle_spread) & (data_angle <= average_angle + angle_spread)
    results = data[mask]

    # Handle if query does not return results
    if len(results) == 0:
        print("No results found for this query.")
        continue

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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    fig.tight_layout()

    # Set ax1 to be the image.
    ax1.imshow(convert)
    ax1.set_title("Input Image")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True)

    # Set ax2 to be the output image.
    ax2.imshow(output)
    ax2.set_title("Output Image")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)

    ax3.autoscale(enable=False, axis='x')
    ax3.autoscale(enable=False, axis='y')

    # Update min_distance and max_distance to consider pixel and pnp approaches.
    min_distance = min(min_distance, distance, pnp_distance)
    max_distance = max(max_distance, distance, pnp_distance)
    min_angle = min(min_angle, inclination_angle, tilt_angle)
    max_angle = max(max_angle, inclination_angle, tilt_angle)

    min_distance -= 5
    max_distance += 5
    min_angle -= 5
    max_angle += 5

    scatter = ax3.scatter(results[:, 0], results[:, 1], c=results[:, 2], label='pixel_count', cmap='coolwarm')
    fig.colorbar(scatter, ax=ax3, location='right', anchor=(0, 0), shrink=0.7)
    ax3.plot([min_distance, max_distance], [inclination_angle, inclination_angle], label='inclination_angle', linestyle='-', color='orange', linewidth=1)
    ax3.plot([distance, distance], [min_angle, max_angle], label='pixel_distance', linestyle='-', color='orange', linewidth=1)
    ax3.plot([min_distance, max_distance], [tilt_angle, tilt_angle], label='tilt_angle', linestyle='-', color='blue', linewidth=1)
    ax3.plot([pnp_distance, pnp_distance], [min_angle, max_angle], label='pnp_distance', linestyle='-', color='blue', linewidth=1)
    ax3.plot([closest_distance, closest_distance], [min_angle, max_angle], label='queried_distance', linestyle='-', color='green')
    ax3.plot([min_distance, max_distance], [closest_angle, closest_angle], label='queried_angle', linestyle='-', color='green')
    ax3.set_xlabel("Distance (mm)")
    ax3.set_ylabel("Angle (degrees)")
    ax3.set_xlim([min_distance, max_distance])
    ax3.set_ylim([min_angle, max_angle])
    ax3.set_title(f"Pixel Count {white_pixels} pixel_spread {pixel_spread} angle_spread {angle_spread} bounds Distance")
    ax3.legend(loc='lower right')
    ax3.grid(True)

    # Save plot.
    plt.savefig(f"/data/irl-results/{render}")
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
