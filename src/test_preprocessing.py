import os
import cv2
from preprocessing import preprocess_image

folder = "dataset/DevanagariHandwrittenCharacterDataset/Images/character_01_ka"

# pick first image automatically
file = os.listdir(folder)[0]

image_path = os.path.join(folder, file)

img = cv2.imread(image_path)

processed = preprocess_image(img)

print("Loaded image:", file)
print("Processed shape:", processed.shape)