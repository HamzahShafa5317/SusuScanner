import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load model
model = joblib.load("model/model_susu.pkl")  # Pastikan file ini ada di folder yang sama

# Konfigurasi halaman
st.set_page_config(page_title="Milk Scanner", layout="centered")

# Judul dan deskripsi
st.markdown("<h1 style='text-align:center; color:#185a9d;'>🐄 Milk Scanner - Prediksi Kualitas Susu</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Unggah gambar susu atau kemasan dan dapatkan prediksi kualitas secara otomatis berdasarkan model machine learning!</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload gambar
uploaded_file = st.file_uploader("📷 Unggah gambar susu atau kemasan:", type=["jpg", "jpeg", "png"])

# Prediksi
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((64, 64)).convert("L")  # grayscale
    st.image(image, caption="🖼️ Gambar yang Diunggah", width=300)

    # Preprocessing
    img_array = np.array(image).flatten().reshape(1, -1)  # Sesuai bentuk input model

    # Prediksi
    prediction = model.predict(img_array)[0]

    # Penjelasan hasil
    details = {
        "susu segar": {
            "text": "✅ Susu terlihat putih cerah, tidak menggumpal, dan berbau segar.",
            "img": "https://images.unsplash.com/photo-1582719478185-2c9c6fcd87c0?auto=format&fit=crop&w=800&q=80"
        },
        "susu basi": {
            "text": "⚠️ Susu tampak menggumpal atau berbau asam, menandakan basi.",
            "img": "https://images.unsplash.com/photo-1585238342028-3c5632dfb035?auto=format&fit=crop&w=800&q=80"
        },
        "kemasan bagus": {
            "text": "✅ Kemasan tampak utuh, bersih, dan tidak ada kerusakan fisik.",
            "img": "https://images.unsplash.com/photo-1606755962773-0f1d0e124b5c?auto=format&fit=crop&w=800&q=80"
        },
        "kemasan rusak": {
            "text": "⚠️ Kemasan terlihat penyok, bocor, atau kotor.",
            "img": "https://images.unsplash.com/photo-1582552488181-d5d6c822ad33?auto=format&fit=crop&w=800&q=80"
        }
    }

    if prediction in details:
        st.markdown("### 📊 Hasil Prediksi")
        st.image(details[prediction]["img"], caption=f"Hasil: {prediction.upper()}", width=400)
        st.success(details[prediction]["text"])
    else:
        st.warning("⚠️ Label tidak dikenali oleh sistem.")
