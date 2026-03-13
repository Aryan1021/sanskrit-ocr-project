import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 32

model = load_model("models/cnn_model.h5")

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

class_folders = list(CHARACTER_MAP.keys())


def preprocess_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray,(IMG_SIZE,IMG_SIZE))

    normalized = resized / 255.0

    reshaped = normalized.reshape(1,32,32,1)

    return reshaped


st.title("Sanskrit OCR using Deep Learning")

uploaded_file = st.file_uploader("Upload a Sanskrit character image")

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image")

    processed = preprocess_image(img)

    prediction = model.predict(processed)

    predicted_index = np.argmax(prediction)

    predicted_folder = class_folders[predicted_index]

    predicted_char = CHARACTER_MAP[predicted_folder]

    st.subheader(f"Predicted Character: {predicted_char}")