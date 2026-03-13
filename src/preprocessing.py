import cv2
import numpy as np

IMG_SIZE = 32


def preprocess_image(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # Resize image
    resized = cv2.resize(blur, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values
    normalized = resized / 255.0

    # Reshape for CNN
    reshaped = normalized.reshape(IMG_SIZE, IMG_SIZE, 1)

    return reshaped