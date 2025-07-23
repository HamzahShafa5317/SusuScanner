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

# Fungsi untuk ekstraksi fitur HOG
def extract_features(image_pil):
    fixed_size = (128, 128)
    image_resized = image_pil.resize(fixed_size)
    image_array = np.array(image_resized)

    # Ubah ke grayscale jika RGB
    if len(image_array.shape) == 3:
        gray = rgb2gray(image_array)
    else:
        gray = image_array

    # Ekstrak fitur HOG tanpa multichannel
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   visualize=False)
    return features

# Fungsi deskripsi hasil prediksi
def get_keterangan(prediksi_label):
    keterangan_dict = {
        "baik": "Susu terdeteksi dalam kondisi **baik**, warna dan bentuk kemasan sesuai standar.",
        "buruk": "Susu kemungkinan dalam kondisi **buruk**, mungkin disebabkan oleh kerusakan kemasan, perubahan warna, atau faktor lainnya.",
        "rusak": "Susu **rusak**, segera periksa kondisi fisik kemasan dan isinya. Tidak disarankan untuk dikonsumsi.",
    }
    return keterangan_dict.get(prediksi_label.lower(), "Tidak diketahui kondisi susu.")

# UI Streamlit
st.set_page_config(page_title="Deteksi Kualitas Susu", layout="centered")
st.title("🥛 Deteksi Kualitas Susu dari Gambar")
st.write("Upload gambar susu untuk diprediksi kualitasnya menggunakan model SVM.")

uploaded_file = st.file_uploader("📤 Pilih file gambar (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar yang diupload", width=250)

        features = extract_features(image)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        st.success(f"✅ Prediksi: **{prediction.upper()}**")
        st.markdown(f"📌 Penjelasan: {get_keterangan(prediction)}")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")
