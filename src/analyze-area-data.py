import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

OUTPUT_PATH = "/data/area_data.csv"
FOCAL_LENGTH = 50
KNOWN_WIDTH = 20
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
SENSOR_WIDTH = 36
SENSOR_HEIGHT = 24
ANGLES = range(0, 68, 1)
DISTANCES = range(105, 206, 1)

FILES = os.listdir('/data/renders')

with open(OUTPUT_PATH, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['distance', 'angle', 'area'])

    for file in FILES:

        # Get the distance and angle from the filename.
        distance = file.split('_')[1]
        angle = file.split('_')[2].split('.')[0]

        img = cv2.imread(f'/data/renders/{file}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        white_pixels = np.sum(thresh == 255)
        writer.writerow([distance, angle, white_pixels])
        f.flush()
