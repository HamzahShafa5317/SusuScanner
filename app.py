import streamlit as st
from PIL import Image
import numpy as np
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

# Load model dan scaler
model_bundle = joblib.load("model_susu.pkl")
model = model_bundle["model"]
scaler = model_bundle["scaler"]

# Fungsi ekstraksi fitur dari gambar
def extract_features(image_pil):
    # Resize ke ukuran yang sesuai pelatihan model (ubah jika model latih pakai ukuran lain)
    fixed_size = (128, 128)
    image_resized = image_pil.resize(fixed_size)

    # Ubah ke array numpy
    image_array = np.array(image_resized)

    # Konversi ke grayscale jika masih RGB
    if len(image_array.shape) == 3:
        gray = rgb2gray(image_array)
    else:
        gray = image_array

    # Ekstraksi fitur HOG
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   visualize=False,
                   multichannel=False)
    return features

# UI Streamlit
st.title("🥛 Deteksi Kualitas Susu dari Gambar")
st.markdown("Upload gambar susu untuk diprediksi kualitasnya menggunakan model SVM.")

uploaded_file = st.file_uploader("Pilih file gambar (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Tampilkan gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", width=250)

        # Ekstrak dan prediksi
        features = extract_features(image)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        st.success(f"✅ Prediksi: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")
