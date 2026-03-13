# Sanskrit Optical Character Recognition (OCR) using Deep Learning

## 📌 Project Overview

This project implements an **Optical Character Recognition (OCR) system for Sanskrit characters written in the Devanagari script** using **Deep Learning**.

The system uses a **Convolutional Neural Network (CNN)** to recognize handwritten Devanagari characters from images.
The trained model can identify **46 classes including Sanskrit characters and digits** with high accuracy.

A **Streamlit web application** is also developed to allow users to upload an image and obtain the predicted Sanskrit character.

---

## 🎯 Objectives

* Recognize Sanskrit characters written in **Devanagari script**
* Train a **CNN-based deep learning model**
* Evaluate the model using **accuracy, precision, recall, and F1-score**
* Visualize performance using a **confusion matrix**
* Build an interactive **web interface for OCR prediction**

---

## 📂 Project Structure

```text
sanskrit-ocr-project
│
├── app
│   └── app.py                 # Streamlit web application
│
├── src
│   ├── preprocessing.py       # Image preprocessing
│   ├── model.py               # CNN architecture
│   ├── train_model.py         # Model training pipeline
│   ├── evaluate_model.py      # Model evaluation
│   └── predict.py             # OCR prediction script
│
├── models
│   └── cnn_model.h5           # Trained deep learning model
│
├── results
│   └── confusion_matrix.png   # Evaluation visualization
│
├── dataset                    # Devanagari dataset (download separately from Kaggle)
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 📊 Dataset

The model is trained on the **Devanagari Handwritten Character Dataset**.

Dataset includes:

* **92,000+ images**
* **46 classes**
* Devanagari characters and digits

Examples of characters:

```
क ख ग घ ङ
च छ ज झ ञ
ट ठ ड ढ ण
त थ द ध न
प फ ब भ म
य र ल व
श ष स ह
क्ष त्र ज्ञ
० १ २ ३ ४ ५ ६ ७ ८ ९
```

Dataset source:

https://www.kaggle.com/datasets/rishianand/devanagari-character-set

---

## 🧠 Model Architecture

The OCR model uses a **Convolutional Neural Network (CNN)**.

Architecture:

```
Input Image (32 × 32 × 1)
        ↓
Conv2D (32 filters)
        ↓
MaxPooling
        ↓
Conv2D (64 filters)
        ↓
MaxPooling
        ↓
Conv2D (128 filters)
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense Layer (256)
        ↓
Dropout
        ↓
Softmax Output Layer (46 classes)
```

---

## 📈 Model Performance

Evaluation metrics:

| Metric    | Score     |
| --------- | --------- |
| Accuracy  | **≈ 98%** |
| Precision | ≈ 0.98    |
| Recall    | ≈ 0.98    |
| F1-score  | ≈ 0.98    |

The confusion matrix indicates **minimal misclassification among visually similar characters**.

---

## 📉 Confusion Matrix

The confusion matrix visualizes classification performance across all 46 classes.

File location:

```
results/confusion_matrix.png
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sanskrit-ocr-project.git
cd sanskrit-ocr-project
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit Web App

```bash
streamlit run app/app.py
```

---

## 🖥️ Application Workflow

```
Upload Image
     ↓
Image Preprocessing
     ↓
CNN Model Prediction
     ↓
Display Predicted Sanskrit Character
```

---

## 🛠 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy**
* **Scikit-learn**
* **Matplotlib / Seaborn**
* **Streamlit**

---

## 🔮 Future Improvements

Possible extensions for this project:

* Word-level Sanskrit OCR
* Text segmentation for full sentences
* Transformer-based OCR models
* Mobile OCR application

---

## 👨‍💻 Author

**Aryan Raj**