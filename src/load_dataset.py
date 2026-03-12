import os
import cv2
import numpy as np

DATASET_PATH = "dataset/DevanagariHandwrittenCharacterDataset/Images"
IMG_SIZE = 32

def load_dataset():

    images = []
    labels = []
    class_names = []

    for folder in os.listdir(DATASET_PATH):

        class_names.append(folder)

        folder_path = os.path.join(DATASET_PATH, folder)

        for file in os.listdir(folder_path):

            img_path = os.path.join(folder_path, file)

            img = cv2.imread(img_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            images.append(img)

            labels.append(folder)

    images = np.array(images) / 255.0
    images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return images, labels, class_names