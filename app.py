# app.py - VERSI FIXED UNTUK STREAMLIT CLOUD
import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

# Konfigurasi
st.set_page_config(
    page_title="BISINDO Classifier",
    page_icon="✋",
    layout="centered"
)

# Judul
st.title("🤟 Klasifikasi Bahasa Isyarat BISINDO")
st.markdown("Upload gambar tangan untuk mengenali huruf BISINDO")

# Load model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('final_bisindo_model.keras')
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Pastikan file 'final_bisindo_model.keras' ada di repository")
        return None

model = load_model()

# Fungsi preprocessing TANPA OpenCV
def preprocess_image(image):
    """Preprocess image menggunakan PIL"""
    # Convert ke RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize ke 128x128
    img_resized = image.resize((128, 128))
    
    # Convert ke numpy dan normalisasi
    img_array = np.array(img_resized) / 255.0
    
    return img_array

# Fungsi prediksi
def predict_image(image, model):
    img_processed = preprocess_image(image)
    img_input = np.expand_dims(img_processed, axis=0)
    
    predictions = model.predict(img_input, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    letter = chr(65 + predicted_idx)  # 0=A, 1=B, dst.
    
    return letter, confidence, predictions[0]

# UI
uploaded_file = st.file_uploader(
    "Pilih gambar tangan...",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file and model:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Gambar Uploaded", width=250)
    
    with col2:
        if st.button("🔍 Prediksi", type="primary"):
            with st.spinner("Menganalisis..."):
                letter, confidence, all_probs = predict_image(image, model)
                
                # Tampilkan hasil
                st.subheader("🎯 Hasil Prediksi")
                
                # Huruf besar
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; 
                background-color: #4CAF50; color: white; 
                border-radius: 15px; margin: 20px 0;">
                    <h1 style="font-size: 96px; margin: 0;">{letter}</h1>
                    <p style="font-size: 24px;">Huruf Prediksi</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence
                st.metric("Confidence", f"{confidence:.2f}%")
                
                # Progress bar
                st.progress(int(confidence) / 100)
                
                if confidence > 90:
                    st.success("✅ Prediksi sangat yakin!")
                elif confidence > 70:
                    st.warning("⚠️ Prediksi cukup yakin")
                else:
                    st.error("❌ Confidence rendah")
                
                # Tampilkan top 5 predictions
                st.subheader("🏆 Top 5 Predictions")
                top_5_idx = np.argsort(all_probs)[-5:][::-1]
                
                for i, idx in enumerate(top_5_idx):
                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        st.markdown(f"**{chr(65 + idx)}**")
                    with col_b:
                        prob = all_probs[idx] * 100
                        st.progress(float(prob/100), text=f"{prob:.1f}%")

# Footer
st.markdown("---")
st.caption("BISINDO Sign Language Classifier | Deployed on Streamlit Cloud")