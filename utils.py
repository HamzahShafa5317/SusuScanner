import numpy as np
from PIL import Image

def load_model_dummy():
    return "dummy_model"

def classify_image(image: Image.Image, model):
    """
    Dummy classifier untuk membedakan gambar susu vs kemasan.
    Berdasarkan kecerahan gambar (grayscale mean pixel).
    """
    image = image.resize((64, 64)).convert("L")
    mean_pixel = np.array(image).mean()
    if mean_pixel > 127:
        return "susu"
    else:
        return "kemasan"

def predict_quality(category: str):
    """
    Prediksi kualitas tergantung dari kategori:
    - Jika 'susu' → susu segar atau susu basi
    - Jika 'kemasan' → kemasan bagus atau kemasan rusak
    """
    if category == "susu":
        label = 'susu segar' if np.random.rand() > 0.5 else 'susu basi'
        details = {
            'susu segar': 'Susu terlihat putih cerah, tidak menggumpal, dan berbau segar.',
            'susu basi': 'Susu tampak menggumpal atau berbau asam, menandakan basi.'
        }
    else:
        label = 'kemasan bagus' if np.random.rand() > 0.5 else 'kemasan rusak'
        details = {
            'kemasan bagus': 'Kemasan tampak utuh, bersih, dan tidak ada kerusakan fisik.',
            'kemasan rusak': 'Kemasan terlihat penyok, bocor, atau kotor.'
        }

    return label, details[label]
