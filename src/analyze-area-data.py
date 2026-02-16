import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

OUTPUT_PATH = "/data/area_data_v2.csv"
FOCAL_LENGTH = 50
KNOWN_WIDTH = 20
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
SENSOR_WIDTH = 36
SENSOR_HEIGHT = 24
ANGLES = range(0, 68, 1)
DISTANCES = range(105, 1006, 1)

FILES = os.listdir('/data/renders')

with open(OUTPUT_PATH, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['known_distance', 'known_angle', 'pixel_count'])

    for file in tqdm(FILES):

        # Get the distance and angle from the filename.
        parts = file.split('_')
        known_distance = parts[1]
        known_angle = parts[2].split('.')[0]

        # Calculate the number of white pixels in the image.
        img = cv2.imread(f'/data/renders/{file}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        white_pixels = np.sum(thresh == 255)

        # Write the data point to the CSV file.
        writer.writerow([int(known_distance), int(known_angle), white_pixels])
