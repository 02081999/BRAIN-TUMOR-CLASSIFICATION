

#py -3.10 --version
#py -3.10 -m venv tf-env
#py -3.10 -m venv tf310-env
#Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
#.\tf-env\Scripts\activate
#pip install tensorflow==2.15.0


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor', 'No Tumor']
MODEL_PATH = r'C:\Users\user\Music\PROJECT5\BRAIN_TUMOR\models\inceptionv3_model.keras'  



st.set_page_config(page_title="Brain MRI Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to predict the tumor type.")

# Upload image
uploaded_file = st.file_uploader("Select an MRI image", type=["jpg", "jpeg", "png"])
#model = load_model(MODEL_PATH, compile=False)
#st.title("Brain Tumor Classification")


if uploaded_file is not None:
    try:
        # ✅ Load model only once and cache it
        @st.cache_resource
        def load_model():
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
            return tf.keras.models.load_model(MODEL_PATH)

        model = load_model()

        # ✅ Show uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # ✅ Preprocess
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # ✅ Predict
        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # ✅ Output
        st.success(f"🎯 Predicted Tumor Type: **{predicted_class.upper()}**")
        st.info(f"📊 Confidence Score: **{confidence * 100:.2f}%**")

    except Exception as e:
        st.error(f"❌ Error: {e}")