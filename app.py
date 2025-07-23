import streamlit as st
from PIL import Image
import numpy as np
import random

st.set_page_config(page_title="Milk Scanner", layout="centered")

# ==== UI ====
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1583324113626-70df0f4deaab?auto=format&fit=crop&w=1470&q=80');
        background-size: cover;
    }
    .title {
        text-align: center;
        font-size: 3em;
        color: #185a9d;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🐮 Milk Scanner - Prediksi Kualitas Susu</div>", unsafe_allow_html=True)
st.markdown("Unggah gambar susu atau kemasan untuk mendapatkan prediksi kualitas!")

# ==== Upload Image ====
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang Diunggah", use_column_width=True)

    # ==== Dummy Deteksi Kemasan vs Susu ====
    image_array = np.array(image.resize((64, 64)).convert("L"))
    mean_pixel = image_array.mean()

    if mean_pixel > 127:
        category = "susu"
        label = "susu segar" if random.random() > 0.5 else "susu basi"
        details = {
            "susu segar": "Susu terlihat putih cerah, tidak menggumpal, dan berbau segar.",
            "susu basi": "Susu tampak menggumpal atau berbau asam, menandakan basi."
        }
        result_image = {
            "susu segar": "https://i.imgur.com/zJ2nN6k.png",
            "susu basi": "https://i.imgur.com/yvBlBNd.png"
        }[label]
    else:
        category = "kemasan"
        label = "kemasan bagus" if random.random() > 0.5 else "kemasan rusak"
        details = {
            "kemasan bagus": "Kemasan tampak utuh, bersih, dan tidak ada kerusakan fisik.",
            "kemasan rusak": "Kemasan terlihat penyok, bocor, atau kotor."
        }
        result_image = {
            "kemasan bagus": "https://i.imgur.com/NnTb5Qf.png",
            "kemasan rusak": "https://i.imgur.com/fcNN2pz.png"
        }[label]

    # ==== Tampilkan Hasil ====
    st.markdown(f"### 📊 Hasil Prediksi: {label.upper()}")
    st.image(result_image, caption=f"Hasil: {label}", use_column_width=True)
    st.markdown(f"📝 **Penjelasan:** {details[label]}")
