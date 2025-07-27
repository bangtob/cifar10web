import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Cache model agar hanya dimuat sekali untuk efisiensi
@st.cache_resource
def load_cnn_model():
    return load_model('model_final1.h5')

model = load_cnn_model()

# Daftar label CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Pengaturan halaman
st.set_page_config(page_title="Klasifikasi Gambar CIFAR-10", layout="centered")
st.title("📷 Aplikasi Klasifikasi Gambar - CNN (New)")
st.markdown("""
Aplikasi ini menggunakan model CNN untuk mengklasifikasikan gambar ke dalam 10 kategori CIFAR-10.
Silakan unggah gambar (ukuran mendekati 32x32 piksel). Model akan memproses dan menampilkan hasil klasifikasi.
""")

# Fungsi preprocessing gambar
def preprocess_image(img: Image.Image) -> np.ndarray:
    img_resized = img.resize((32, 32))
    img_array = np.array(img_resized).astype('float32') / 255.0
    if img_array.shape != (32, 32, 3):
        raise ValueError("Gambar harus memiliki 3 kanal warna (RGB).")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_file = st.file_uploader("Unggah gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Buka dan tampilkan gambar yang diupload
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='🖼️ Gambar yang diunggah', use_column_width=True)

        # Preprocessing gambar
        img_array = preprocess_image(img)

        # Tombol prediksi
        if st.button("🔍 Prediksi Gambar"):
            prediction = model.predict(img_array)
            predicted_idx = np.argmax(prediction)
            predicted_class = class_names[predicted_idx]
            confidence = prediction[0][predicted_idx] * 100

            st.success(f"Hasil Prediksi: **{predicted_class}**")
            st.info(f"Tingkat Keyakinan: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

else:
    st.info("Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
