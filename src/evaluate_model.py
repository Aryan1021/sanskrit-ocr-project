import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

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

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        images.append(img)

        labels.append(folder)

images = np.array(images) / 255.0
images = images.reshape(-1,32,32,1)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

labels_cat = to_categorical(labels_encoded)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels_cat,
    test_size=0.2,
    random_state=42
)

print("Dataset ready")

# Load trained model
model = load_model("models/cnn_model.h5")

# Predict
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
print("\nClassification Report:\n")

print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("results/confusion_matrix.png")

print("\nConfusion matrix saved in results/")