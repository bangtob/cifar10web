import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model CNN
@st.cache_resource
def load_cnn_model():
    return load_model('model_final1.h5')

model = load_cnn_model()

# Daftar label CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Judul aplikasi
st.set_page_config(page_title="Klasifikasi Gambar CIFAR-10", layout="centered")
st.title("📷 Aplikasi Klasifikasi Gambar - CNN")
st.markdown("""
Aplikasi ini menggunakan model CNN untuk mengklasifikasikan gambar ke dalam 10 kategori CIFAR-10.
Silakan unggah gambar (ukuran mendekati 32x32 piksel). Model akan memproses dan menampilkan hasil klasifikasi.
""")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Tampilkan gambar
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='🖼️ Gambar yang diunggah', use_column_width=True)

        # Preprocessing
        img_resized = img.resize((32, 32))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        if st.button("🔍 Prediksi Gambar"):
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.success(f"Hasil Prediksi: **{predicted_class}**")
            st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
