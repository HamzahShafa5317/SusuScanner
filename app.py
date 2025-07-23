import streamlit as st
from PIL import Image
import numpy as np
import pickle
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog

# Load model
with open("model_susu.pkl", "rb") as f:
    bundle = pickle.load(f)
    scaler = bundle["scaler"]
    selector = bundle["selector"]
    model = bundle["model"]

# UI Title
st.set_page_config(layout="wide")
st.title("🥛 Prediksi Jenis Susu")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("susu.png", caption="Gambar Susu", use_column_width=True)

with col2:
    st.markdown("### 📤 Upload Gambar Susu")
    uploaded_file = st.file_uploader("Pilih file gambar (.jpg/.png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang Diunggah", width=300)

        # Preprocess image
        img_resized = img.resize((128, 128))
        gray = rgb2gray(np.array(img_resized))
        features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, multichannel=False)

        # Transform with scaler & selector
        X_scaled = scaler.transform([features])
        X_selected = selector.transform(X_scaled)

        # Predict
        prediction = model.predict(X_selected)[0]
        st.success(f"✅ Prediksi: **{prediction}**")
