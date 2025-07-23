import streamlit as st
from PIL import Image
from utils import load_model_dummy, classify_image, predict_quality

st.set_page_config(page_title="Prediksi Susu atau Kemasan", layout="centered")

st.title("🔍 Prediksi Kualitas Susu atau Kemasan")
st.write("Unggah gambar susu atau kemasannya. Aplikasi akan menentukan jenisnya dan memberikan prediksi kualitas.")

uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    with st.spinner("Menganalisis gambar..."):
        model = load_model_dummy()
        kategori = classify_image(image, model)
        label, detail = predict_quality(kategori)

        st.markdown(f"### Hasil Deteksi: **{kategori.upper()}**")
        st.markdown(f"### Prediksi: 🟢 **{label}**")
        st.info(detail)
