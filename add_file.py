import math
from os import walk
import cv2
import os
from pathlib import Path
from face_functions import speak, add_to_database, add_to_database2

import cv2
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        return images


folder = "C:/Users/ASUS/Documents/aligned_images_DB"

load_images_from_folder(folder)

f = []
for (_, _, filenames) in walk(folder):
    break

for i in range(0, math.ceil(len(filenames) / 100 * 70)):
    add_to_database(filenames[i])

for j in range(i + 1, len(filenames)):
    if filenames[j] != "":
        add_to_database2(filenames[j])
