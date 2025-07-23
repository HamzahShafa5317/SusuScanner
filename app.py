import streamlit as st
import numpy as np
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
import joblib
from PIL import Image
import io

# Load model dan scaler
model_data = joblib.load("model_susu.pkl")
model = model_data["model"]
scaler = model_data["scaler"]

# Mapping label ke deskripsi
label_deskripsi = {
    "kemasan_bagus": "✅ Kemasan dalam kondisi bagus dan rapi. Aman untuk dikonsumsi.",
    "kemasan_rusak": "⚠️ Kemasan rusak. Perlu dicek lebih lanjut karena bisa bocor atau terkontaminasi.",
    "susu_segar": "🥛 Susu masih segar. Warna dan tekstur normal.",
    "susu_basi": "❌ Susu kemungkinan sudah basi. Tidak layak konsumsi."
}

def extract_features(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((128, 128))
    image_np = np.array(image)
    gray = rgb2gray(image_np)
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   visualize=False)
    return features

# UI
st.set_page_config(page_title="Deteksi Kualitas Susu", layout="centered")

st.markdown("<h1 style='text-align: center;'>🥛 Deteksi Kualitas Susu dari Gambar</h1>", unsafe_allow_html=True)
st.markdown("Upload gambar susu untuk diprediksi kualitasnya menggunakan model SVM.")
uploaded_file = st.file_uploader("Pilih file gambar (.jpg/.png)", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        image_bytes = uploaded_file.read()
        features = extract_features(image_bytes)
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]

        st.markdown(f"### 🧠 Prediksi: **{pred.replace('_', ' ').title()}**")
        st.info(label_deskripsi.get(pred, "Label tidak dikenali."))

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")
