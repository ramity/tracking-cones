import cv2
import numpy as np
import math
import os

# Load image
# render_00 = '/data/renders/render_00_110_0.00_-110.00_0.00_1.00_0.00_0.00_1.00_0.00_1.00.png'
# render_05 = '/data/renders/render_05_110_0.00_-109.58_9.59_1.00_0.09_0.00_1.00_0.00_1.00.png'
# render_45 = '/data/renders/render_45_110_0.00_-77.78_77.78_0.71_0.71_0.00_1.00_0.00_1.00.png'
# render_60 = '/data/renders/render_60_110_0.00_-55.00_95.26_0.50_0.87_0.00_1.00_0.00_1.00.png'
# render_67 = '/data/renders/render_67_110_0.00_-42.98_101.26_0.39_0.92_0.00_1.00_0.00_1.00.png'
# render_68 = '/data/renders/render_68_110_0.00_-41.21_101.99_0.37_0.93_0.00_1.00_0.00_1.00.png'
# img = cv2.imread(render_60)

render_directory = '/data/renders_400/'

# For each render
for i, render in enumerate(os.listdir(render_directory)):
    img = cv2.imread(os.path.join(render_directory, render))
    original = img.copy()

    output_file = '/data/contours_400/' + render

    # Extract contours from image.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw extracted contours.
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    cv2.imwrite(output_file, img)

    # Get the top most y value.
    top_y = 1080
    for contour in contours[0]:
        if contour[0, 1] < top_y:
            top_y = contour[0, 1]

    # print(top_y)

    # Get the number of points that are equal to the top_y value.
    top_most_contours = [c for c in contours[0] if c[0, 1] == top_y]
    cv2.drawContours(img, top_most_contours, -1, (0, 255, 0), 1)
    cv2.imwrite(output_file, img)

    # print(len(top_most_contours))

    # Sometimes there are multiple contours at the tip, so we need to find the left and right tip.
    left_tip = sorted(top_most_contours, key=lambda c: c[0, 0])[0]
    right_tip = sorted(top_most_contours, key=lambda c: c[0, 0], reverse=True)[0]

    # print(left_tip)
    # print(right_tip)

    # Get all points with x values less than the left tip and greater than the right tip.
    left_canidates = [c for c in contours[0] if c[0, 0] < left_tip[0, 0]]
    right_canidates = [c for c in contours[0] if c[0, 0] > right_tip[0, 0]]

    # Calculate slopes for left canidates.
    left_slopes = []
    for c in left_canidates:
        slope = (c[0, 1] - left_tip[0, 1]) / (c[0, 0] - left_tip[0, 0])
        left_slopes.append(slope)

    # Calculate slopes for right canidates.
    right_slopes = []
    for c in right_canidates:
        slope = (c[0, 1] - right_tip[0, 1]) / (c[0, 0] - right_tip[0, 0])
        right_slopes.append(slope)

    # print(left_slopes)
    # print(right_slopes)

    # Calculate the unique slopes and their frequency for left and right canidates.
    unique_left_slopes, left_slope_counts = np.unique(left_slopes, return_counts=True)
    unique_right_slopes, right_slope_counts = np.unique(right_slopes, return_counts=True)

    # print(unique_left_slopes)
    # print(unique_right_slopes)
    # print(left_slope_counts)
    # print(right_slope_counts)

    # Calculate slopes by using the most common slope.
    left_slope = unique_left_slopes[np.argmax(left_slope_counts)]
    right_slope = unique_right_slopes[np.argmax(right_slope_counts)]

    # print(left_slope)
    # print(right_slope)

    # Alternative approach: Calculate slopes by using the weighted average of the slopes.
    # left_slope = np.average(unique_left_slopes, weights=left_slope_counts)
    # right_slope = np.average(unique_right_slopes, weights=right_slope_counts)

    # Draw a line from left_tip - 1000 units to left_tip left_slope
    cv2.line(img, np.int32(left_tip[0] - (1000, (1000 * left_slope))), np.int32(left_tip[0]), (0, 255, 0), 1)

    # Draw a line from right_tip + 1000 units to right_tip right_slope
    cv2.line(img, np.int32(right_tip[0] + (1000, (1000 * right_slope))), np.int32(right_tip[0]), (0, 255, 0), 1)
    cv2.imwrite(output_file, img)

    # Get Left most countours from left_canidates.
    left_most_x = left_tip[0, 0]
    for c in left_canidates:
        if c[0, 0] < left_most_x:
            left_most_x = c[0, 0]

    # print(left_most_x)

    # Get Right most countours from right_canidates.
    right_most_x = right_tip[0, 0]
    for c in right_canidates:
        if c[0, 0] > right_most_x:
            right_most_x = c[0, 0]

    # print(right_most_x)

    # Get left most countours.
    left_most_contours = [c for c in left_canidates if c[0, 0] == left_most_x]

    # Get right most countours.
    right_most_contours = [c for c in right_canidates if c[0, 0] == right_most_x]

    # Draw left most countours.
    cv2.drawContours(img, left_most_contours, -1, (0, 255, 0), 1)

    # Draw right most countours.
    cv2.drawContours(img, right_most_contours, -1, (0, 255, 0), 1)
    cv2.imwrite(output_file, img)

    # print(left_most_contours)
    # print(right_most_contours)

    # Get y center of left most countours.
    left_most_y_center = (np.mean([c[0, 1] for c in left_most_contours]))

    # Get y center of right most countours.
    right_most_y_center = (np.mean([c[0, 1] for c in right_most_contours]))

    # print(left_most_y_center)
    # print(right_most_y_center)

    base_left = (left_most_x, left_most_y_center)
    base_right = (right_most_x, right_most_y_center)

    # Draw a line from left_most_x, left_most_y_center to right_most_x, right_most_y_center.
    cv2.line(img, np.int32(base_left), np.int32(base_right), (0, 255, 0), 1)
    cv2.imwrite(output_file, img)

    # Calculate the center of the line.
    base_center_x = ((left_most_x + right_most_x) / 2)
    base_center_y = ((left_most_y_center + right_most_y_center) / 2)
    base_center = (base_center_x, base_center_y)

    # Calculate the center of the left and right tips.
    tip_center_x = ((left_tip[0, 0] + right_tip[0, 0]) / 2)
    tip_center_y = ((left_tip[0, 1] + right_tip[0, 1]) / 2)
    tip_center = (tip_center_x, tip_center_y)

    # Draw a line from the base_center to center_tip.
    cv2.line(img, np.int32(base_center), np.int32(tip_center), (0, 255, 0), 1)
    cv2.imwrite(output_file, img)

    # Calculate the angle of the line.
    base_center_line_rad = math.atan2(tip_center_y - base_center_y, tip_center_x - base_center_x)

    # Calculate the slope between left_tip and base_left.
    base_left_slope = (left_tip[0, 1] - base_left[1]) / (left_tip[0, 0] - base_left[0])

    # Calculate the slope between right_tip and base_right.
    base_right_slope = (right_tip[0, 1] - base_right[1]) / (right_tip[0, 0] - base_right[0])

    # print(base_left_slope)
    # print(base_right_slope)

    left_slope_delta = abs(base_left_slope - left_slope)
    right_slope_delta = abs(base_right_slope - right_slope)

    # print(left_slope_delta)
    # print(right_slope_delta)

    # Calculate the bottom_y value
    bottom_y = 0
    for c in contours[0]:
        if c[0, 1] > bottom_y:
            bottom_y = c[0, 1]

    # print(bottom_y)

    # Get countours which occur at the bottom_y value.
    bottom_most_contours = [c for c in contours[0] if c[0, 1] == bottom_y]
    # cv2.drawContours(img, bottom_most_contours, -1, (255, 0, 0), 1)
    cv2.imwrite(output_file, img)

    # Get the left most and right most contours from the bottom_most_contours.
    left_most_bottom_contour = min(bottom_most_contours, key=lambda c: c[0, 0])
    right_most_bottom_contour = max(bottom_most_contours, key=lambda c: c[0, 0])

    # print(left_most_bottom_contour)
    # print(right_most_bottom_contour)

    cv2.drawContours(img, [left_most_bottom_contour], -1, (0, 255, 0), 1)
    cv2.drawContours(img, [right_most_bottom_contour], -1, (0, 255, 0), 1)
    cv2.imwrite(output_file, img)

    # Get x center between left_most_bottom_contour and right_most_bottom_contour.
    bottom_center_x = ((left_most_bottom_contour[0, 0] + right_most_bottom_contour[0, 0]) / 2)
    bottom_center = (bottom_center_x, bottom_y)

    # Draw a line from the base_center to tip_center.
    cv2.line(img, np.int32(base_center), np.int32(tip_center), (0, 255, 0), 1)
    cv2.imwrite(output_file, img)

    # Draw a line from the base_center to bottom_center.
    cv2.line(img, np.int32(base_center), np.int32(bottom_center), (0, 255, 0), 1)
    cv2.imwrite(output_file, img)

    # Draw an ellipse using:
    # - the base_center as the center
    # - its distance to the bottom_center as the minor axis.
    # - base_left to base_right as the major axis.
    axes = (abs(base_center[0] - base_left[0]), abs(base_center[1] - bottom_center[1]))
    cv2.ellipse(img, np.int32(base_center), np.int32(axes), 0, 0, 360, (0, 255, 0), 1)
    cv2.imwrite(output_file, img)

    # Assuming default focal length of 50mm, calculate the distance from the camera to the cone.
    focal_length = 50
    known_width = 20
    sensor_width = 36
    render_width = 1920
    known_distance = 400

    # Get base distance between points base_left and base_right.
    base_distance = math.sqrt((base_left[0] - base_right[0]) ** 2 + (base_left[1] - base_right[1]) ** 2)
    # print(base_distance)

    # Calculate the distance from the camera to the cone.
    distance = (known_width * focal_length) / ((base_distance / render_width) * sensor_width)
    angle_to_cone = math.degrees(math.asin(axes[1] / axes[0]))
    distance_error = f"{((abs(distance - known_distance) / known_distance) * 100):.2f}%"

    if i:
        angle_error = f"{((abs(angle_to_cone - i) / i) * 100):.2f}%"
    else:
        angle_error = f"{abs(angle_to_cone):.2f}%"

    # Calculate distance from tip to base
    tip_to_base_distance = math.sqrt((tip_center[0] - base_center[0]) ** 2 + (tip_center[1] - base_center[1]) ** 2)
    # print(i, distance, angle_to_cone, tip_to_base_distance, tip_to_base_distance / base_distance)

    # Calculate the number of white pixels in the image.
    white_pixels = np.sum(original > 0)
    white_pixels_percentage = white_pixels / (original.shape[0] * original.shape[1]) * 100
    min_percentage = 25.902922453703702
    max_percentage = 26.421006944444443
    normalized_white_pixels = (white_pixels_percentage - min_percentage) / (max_percentage - min_percentage)
    print(i, white_pixels, normalized_white_pixels)
