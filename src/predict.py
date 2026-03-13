import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 32
DATASET_PATH = "dataset/DevanagariHandwrittenCharacterDataset/Images"

# Load trained model
model = load_model("models/cnn_model.h5")

# Character mapping
CHARACTER_MAP = {
"character_01_ka": "क",
"character_02_kha": "ख",
"character_03_ga": "ग",
"character_04_gha": "घ",
"character_05_kna": "ङ",
"character_06_cha": "च",
"character_07_chha": "छ",
"character_08_ja": "ज",
"character_09_jha": "झ",
"character_10_yna": "ञ",
"character_11_taamatar": "ट",
"character_12_thaa": "ठ",
"character_13_daa": "ड",
"character_14_dhaa": "ढ",
"character_15_adna": "ण",
"character_16_tabala": "त",
"character_17_tha": "थ",
"character_18_da": "द",
"character_19_dha": "ध",
"character_20_na": "न",
"character_21_pa": "प",
"character_22_pha": "फ",
"character_23_ba": "ब",
"character_24_bha": "भ",
"character_25_ma": "म",
"character_26_yaw": "य",
"character_27_ra": "र",
"character_28_la": "ल",
"character_29_waw": "व",
"character_30_motosaw": "श",
"character_31_petchiryakha": "ष",
"character_32_patalosaw": "स",
"character_33_ha": "ह",
"character_34_chhya": "क्ष",
"character_35_tra": "त्र",
"character_36_gya": "ज्ञ",

# digits
"digit_0": "०",
"digit_1": "१",
"digit_2": "२",
"digit_3": "३",
"digit_4": "४",
"digit_5": "५",
"digit_6": "६",
"digit_7": "७",
"digit_8": "८",
"digit_9": "९"
}


def preprocess_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    normalized = resized / 255.0

    reshaped = normalized.reshape(1,32,32,1)

    return reshaped


def predict_multiple_classes():

    print("\nTesting OCR predictions\n")

    class_folders = sorted(os.listdir(DATASET_PATH))

    for folder in class_folders:

        folder_path = os.path.join(DATASET_PATH, folder)

        file = os.listdir(folder_path)[0]

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        processed = preprocess_image(img)

        prediction = model.predict(processed)

        predicted_index = np.argmax(prediction)

        predicted_folder = class_folders[predicted_index]

        actual_char = CHARACTER_MAP.get(folder, folder)
        predicted_char = CHARACTER_MAP.get(predicted_folder, predicted_folder)

        print(f"Actual: {actual_char} | Predicted: {predicted_char}")
        print("--------------------------------------")


predict_multiple_classes()