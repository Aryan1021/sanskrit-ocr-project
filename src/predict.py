import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 32

# Load trained model
model = load_model("models/cnn_model.h5")

# Character mapping (index → character)
CHARACTER_MAP = [
'क','ख','ग','घ','ङ',
'च','छ','ज','झ','ञ',
'ट','ठ','ड','ढ','ण',
'त','थ','द','ध','न',
'प','फ','ब','भ','म',
'य','र','ल','व',
'श','ष','स','ह',
'क्ष','त्र','ज्ञ'
]


def preprocess_image(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray,(IMG_SIZE,IMG_SIZE))

    normalized = resized / 255.0

    reshaped = normalized.reshape(1,32,32,1)

    return reshaped


def predict_character(image_path):

    img = preprocess_image(image_path)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)

    print("Predicted class index:", class_index)

    if class_index < len(CHARACTER_MAP):
        print("Predicted character:", CHARACTER_MAP[class_index])
    else:
        print("Character index:", class_index)


# Test image
predict_character("test_images/94928.png")