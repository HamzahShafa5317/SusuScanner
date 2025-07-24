import streamlit as st
from PIL import Image
import numpy as np
import random

# Konfigurasi halaman
st.set_page_config(page_title="Milk Scanner", layout="centered")

# Judul dan deskripsi
st.markdown("<h1 style='text-align:center; color:#185a9d;'>🐄 Milk Scanner - Prediksi Kualitas Susu</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Unggah gambar susu atau kemasan dan dapatkan prediksi kualitas secara otomatis!</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload gambar
uploaded_file = st.file_uploader("📷 Unggah gambar susu atau kemasan:", type=["jpg", "jpeg", "png"])

# Dummy prediksi dan tampilan hasil
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang Diunggah", width=400)

    # Dummy kategori (berbasis mean pixel grayscale)
    img_array = np.array(image.resize((64, 64)).convert("L"))
    mean_pixel = img_array.mean()

    if mean_pixel > 127:
        category = "susu"
        label = "susu segar" if random.random() > 0.5 else "susu basi"
        details = {
            "susu segar": "✅ Susu terlihat putih cerah, tidak menggumpal, dan berbau segar.",
            "susu basi": "⚠️ Susu tampak menggumpal atau berbau asam, menandakan basi."
        }
        result_image = {
            "susu segar": "https://images.unsplash.com/photo-1582719478185-2c9c6fcd87c0?auto=format&fit=crop&w=800&q=80",
            "susu basi": "https://images.unsplash.com/photo-1585238342028-3c5632dfb035?auto=format&fit=crop&w=800&q=80"
        }[label]
    else:
        category = "kemasan"
        label = "kemasan bagus" if random.random() > 0.5 else "kemasan rusak"
        details = {
            "kemasan bagus": "✅ Kemasan tampak utuh, bersih, dan tidak ada kerusakan fisik.",
            "kemasan rusak": "⚠️ Kemasan terlihat penyok, bocor, atau kotor."
        }
        result_image = {
            "kemasan bagus": "https://images.unsplash.com/photo-1606755962773-0f1d0e124b5c?auto=format&fit=crop&w=800&q=80",
            "kemasan rusak": "https://images.unsplash.com/photo-1582552488181-d5d6c822ad33?auto=format&fit=crop&w=800&q=80"
        }[label]

    # Tampilkan hasil
    st.markdown("### 📊 Hasil Prediksi")
    st.image(result_image, caption=f"Hasil: {label.upper()}", width=400)
    st.info(details[label])
