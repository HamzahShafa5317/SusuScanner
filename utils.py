import numpy as np
from PIL import Image

def load_model_dummy():
    return "dummy_model"

def classify_image(image: Image.Image, model):
    image = image.resize((64, 64)).convert("L")
    mean_pixel = np.array(image).mean()

    if mean_pixel > 127:
        return "susu"
    else:
        return "kemasan"

def predict_quality(category: str):
    import random
    if category == "susu":
        labels = ['Bagus', 'Basi', 'Rusak', 'Segar']
        details = {
            'Bagus': 'Warna putih bersih, tidak menggumpal, dan tidak berbau.',
            'Basi': 'Menggumpal, berbau asam, atau warna kekuningan.',
            'Rusak': 'Tampak ada kontaminasi, warna tidak merata.',
            'Segar': 'Kondisi baru, suhu penyimpanan ideal, dan tampilan normal.'
        }
    else:
        labels = ['Utuh', 'Kotor', 'Rusak', 'Bocor']
        details = {
            'Utuh': 'Tidak ada kerusakan fisik pada kemasan.',
            'Kotor': 'Kemasan tampak kotor oleh debu atau noda.',
            'Rusak': 'Terdapat sobekan atau penyok pada kemasan.',
            'Bocor': 'Kemasan tidak kedap, cairan keluar atau merembes.'
        }
    
    label = random.choice(labels)
    return label, details[label]
