import streamlit as st
import numpy as np
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# Update this to match your saved model format
model_path = "plant_disease_model.keras"  # or "plant_disease_model.h5" or "plant_disease_model"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1Sy3VPGK5F502z1YXiZi0SXGQ9KWWEUxs"
    gdown.download(url, model_path, quiet=False)

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Load model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load class labels
with open("label_classes.pkl", "rb") as f:
    class_names = pickle.load(f)

# Page styling
page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background-image: linear-gradient(rgba(200,255,210,0.6), rgba(200,255,210,0.6)),
url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}
[data-testid="stHeader"]{background: rgba(0,0,0,0);}
.title{text-align:center;font-size:52px !important;font-weight:bold;color:#0d4d1c;margin-top:20px;}
.subtitle{text-align:center;font-size:26px !important;color:#1b5e20;margin-bottom:30px;}
.stFileUploader{background: rgba(255,255,255,0.5);backdrop-filter: blur(10px);border:2px dashed #2e7d32;border-radius:15px;padding:20px;text-align:center;font-size:18px;color:#0b3d14;transition:0.3s;}
.stFileUploader:hover{border-color:#1b5e20;background: rgba(255,255,255,0.7);box-shadow:0px 4px 15px rgba(0,0,0,0.2);}
.stFileUploader button{background: linear-gradient(90deg,#2e7d32,#66bb6a);color:white;border:none;border-radius:20px;padding:8px 20px;font-size:16px;}
.result-box{background: rgba(255,255,255,0.4);backdrop-filter: blur(10px);border-radius:15px;padding:25px;font-size:24px;text-align:center;border:1px solid rgba(255,255,255,0.5);color:#0b3d14;margin-top:20px;}
.stButton>button{background: linear-gradient(90deg,#2e7d32,#66bb6a);color:white;border:none;border-radius:25px;padding:12px 30px;font-size:18px;transition:0.3s;}
.stButton>button:hover{transform:scale(1.05);box-shadow:0px 5px 15px rgba(0,0,0,0.3);}
img{border-radius:15px;}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown('<p class="title">🌿 Plant Disease Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a leaf image and detect plant diseases using Artificial Intelligence</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(
        f"""
        <div class="result-box">
        🌱 <b>Predicted Disease:</b> {predicted_class} <br><br>
        📊 <b>Confidence:</b> {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📈 Confidence Level")
    progress_bar = st.progress(0)
    for i in range(int(confidence) + 1):
        progress_bar.progress(i)

    st.markdown(
        f"""
        <div style="
        background: rgba(255,255,255,0.4);
        backdrop-filter: blur(10px);
        padding:10px;
        border-radius:10px;
        text-align:center;
        font-size:20px;
        color:#0b3d14;
        margin-top:10px;
        ">
        <strong>{confidence:.2f}% Confidence</strong>
        </div>
        """,
        unsafe_allow_html=True

    )
