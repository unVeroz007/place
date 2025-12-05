# app.py - Versi SIMPLIFIED
import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image
import os

# Judul
st.title("🤟 Klasifikasi Bahasa Isyarat BISINDO")
st.write("Upload gambar tangan untuk mengenali huruf BISINDO (A-Z)")

# Load model
@st.cache_resource
def load_model():
    try:
        # PASTIKAN MODEL ADA DI FOLDER YANG SAMA
        model = keras.models.load_model('final_bisindo_model.keras')
        return model
    except:
        st.error("Model tidak ditemukan! Pastikan file 'final_bisindo_model.keras' ada di folder yang sama.")
        return None

model = load_model()

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", width=300)
    
    # Tombol prediksi
    if st.button("🔍 Prediksi Sekarang"):
        with st.spinner("Sedang menganalisis..."):
            # Preprocessing
            img_array = np.array(image)
            
            # Convert ke RGB jika perlu
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            # Resize ke 128x128 (sesuai model)
            img_resized = cv2.resize(img_array, (128, 128))
            
            # Normalisasi
            img_input = img_resized / 255.0
            img_input = np.expand_dims(img_input, axis=0)
            
            # Prediksi
            predictions = model.predict(img_input, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx] * 100
            
            # Konversi ke huruf (0=A, 1=B, dst)
            huruf = chr(65 + predicted_idx)
            
            # Tampilkan hasil
            st.success(f"✅ **HASIL PREDIKSI: HURUF '{huruf}'**")
            st.info(f"**Tingkat Kepercayaan: {confidence:.2f}%**")
            
            # Progress bar
            st.progress(int(confidence))
            
            # Jika confidence rendah
            if confidence < 80:
                st.warning("⚠️ Confidence rendah, mungkin hasil kurang akurat")
            
            # Tampilkan top 3
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            st.write("**Top 3 Prediksi:**")
            for i, idx in enumerate(top_3_idx):
                letter = chr(65 + idx)
                prob = predictions[0][idx] * 100
                st.write(f"{i+1}. {letter}: {prob:.1f}%")

st.markdown("---")
st.write("Aplikasi Klasifikasi BISINDO by Anda | Deployed with Streamlit")