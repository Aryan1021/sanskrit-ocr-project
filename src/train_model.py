import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from model import create_model

DATASET_PATH = "dataset/DevanagariHandwrittenCharacterDataset/Images"
IMG_SIZE = 32

images = []
labels = []

print("Loading dataset...")

for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, folder)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        images.append(img)

        labels.append(folder)

images = np.array(images) / 255.0
images = images.reshape(-1, 32, 32, 1)

print("Dataset loaded")

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)

labels = to_categorical(labels)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create model
model = create_model(labels.shape[1])

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# Save model
model.save("models/cnn_model.h5")

print("Model saved successfully")