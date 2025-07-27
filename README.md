# BRAIN-TUMOR-CLASSIFICATION


# 🧠 Brain Tumor Classification Using Deep Learning

This project implements a deep learning pipeline for classifying brain MRI images into four distinct categories: **Glioma Tumor**, **Meningioma Tumor**, **Pituitary Tumor**, and **No Tumor**. It includes both custom CNN and transfer learning models (MobileNetV2, InceptionV3, etc.), evaluated using key performance metrics. The trained models are deployed in a user-friendly **Streamlit** web app for real-time tumor prediction.

---

## 📌 Problem Statement

Detecting brain tumors accurately from MRI images is critical for early diagnosis and treatment planning. Manual diagnosis is prone to human error and subjectivity. This project aims to automate the classification of brain tumors using deep learning techniques, providing fast and accurate predictions to assist radiologists and healthcare professionals.

---

## 📂 Dataset

The dataset used for training and validation includes MRI brain scans categorized into:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**


📝 *Each image is preprocessed, resized to 224x224, normalized, and augmented before training.*

---

## 🛠️ Technologies & Libraries

- **Language**: Python 3.10
- 
- **Libraries**:
- 
  - `TensorFlow`, `Keras` – model training & transfer learning
  - 
  - `NumPy`, `Pandas`, `Matplotlib`, `Seaborn` – data manipulation & visualization
  - 
  - `Streamlit` – web app development
  - 
  - `PIL` – image preprocessing
 
  - 
- **Models**:
- 
  - Custom CNN
  - 
  - MobileNetV2
 
  - InceptionV3
  - 
  - ResNet50
  - 
  - EfficientNetB0

---

## 🔍 Model Architectures

### 1. Custom CNN

Built from scratch with convolutional, pooling, and dense layers.

### 2. Transfer Learning Models

All pretrained on **ImageNet**, with custom classification heads added:

- **MobileNetV2**
- 
- **InceptionV3**
- 
- **ResNet50**
- 
- **EfficientNetB0**

## 🧪 Training & Callbacks

Models were trained using:

EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

ModelCheckpoint(filepath='models/model_name.keras', save_best_only=True)

All .h5 models were later converted to .keras format for TensorFlow 2.11+ compatibility.

## 📊 Visualizations

Training vs Validation Accuracy

Training vs Validation Loss

Confusion Matrix

Classification Report

## 🌐 Streamlit Web App

A minimal web interface built using Streamlit lets users upload brain MRI images and receive real-time tumor type predictions.

🔧 Features:

Upload .jpg, .png, .jpeg MRI images

Real-time classification

Display of predicted class
