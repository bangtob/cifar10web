import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model CNN yang sudah disimpan
model = load_model('model_final.h5')

# Label kelas untuk CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Judul aplikasi
st.title("📷 Aplikasi Klasifikasi Gambar - CNN")

st.markdown("""
Aplikasi ini menggunakan model CNN untuk mengklasifikasikan gambar ke dalam 10 kategori CIFAR-10.
Silakan unggah gambar (ukuran mendekati 32x32 piksel).
""")

# Upload gambar dari user
uploaded_file = st.file_uploader("Unggah gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar yang diunggah', use_column_width=True)

    # Proses gambar: resize, normalisasi, ubah format
    img = img.resize((32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Tombol prediksi
    if st.button("🔍 Prediksi Gambar"):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Hasil Prediksi: **{predicted_class}**")
        st.info(f"Akurasi Keyakinan: **{confidence:.2f}%**")
