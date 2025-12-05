# app.py - FINAL VERSION FOR STREAMLIT CLOUD
import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import tempfile
import os

# ============================================
# KONFIGURASI
# ============================================
st.set_page_config(
    page_title="BISINDO Classifier",
    page_icon="✋",
    layout="centered"
)

# ============================================
# JUDUL APLIKASI
# ============================================
st.title("🤟 Klasifikasi Bahasa Isyarat BISINDO")
st.markdown("""
Upload gambar tangan untuk mengenali huruf BISINDO (A-Z).
Model AI akan memprediksi huruf yang ditunjukkan.
""")

# ============================================
# LOAD MODEL - DENGAN ERROR HANDLING
# ============================================
@st.cache_resource
def load_model():
    """Load model dengan caching untuk performa"""
    try:
        # Coba beberapa lokasi yang mungkin
        possible_paths = [
            'final_bisindo_model.keras',
            './final_bisindo_model.keras',
            '/mount/src/place/final_bisindo_model.keras'
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                st.sidebar.success(f"✅ Model loaded from: {model_path}")
                return model
        
        # Jika tidak ditemukan
        st.sidebar.error("❌ Model file not found!")
        st.sidebar.info("""
        Pastikan file 'final_bisindo_model.keras' ada di repository.
        Upload file ke GitHub repository Anda.
        """)
        return None
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

# ============================================
# FUNGSI PREPROCESSING TANPA OPENCV
# ============================================
def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess image menggunakan PIL saja (tanpa OpenCV)
    
    Args:
        image: PIL Image object
        target_size: Tuple (width, height)
    
    Returns:
        numpy array siap untuk model
    """
    # Convert ke RGB jika bukan RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize ke target size
    img_resized = image.resize(target_size)
    
    # Convert ke numpy array
    img_array = np.array(img_resized)
    
    # Normalisasi ke [0, 1]
    img_normalized = img_array / 255.0
    
    return img_normalized

# ============================================
# FUNGSI PREDIKSI
# ============================================
def predict_image(image, model):
    """
    Predict huruf dari gambar
    
    Returns:
        letter: huruf prediksi (A-Z)
        confidence: confidence score (0-100%)
        all_probs: semua probabilitas
    """
    # Preprocess
    img_processed = preprocess_image(image)
    
    # Add batch dimension
    img_input = np.expand_dims(img_processed, axis=0)
    
    # Predict
    predictions = model.predict(img_input, verbose=0)
    
    # Get results
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    letter = chr(65 + predicted_idx)  # 0->A, 1->B, ..., 25->Z
    
    return letter, confidence, predictions[0]

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    confidence_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=50,
        max_value=99,
        value=80,
        help="Prediksi di bawah threshold akan dianggap kurang yakin"
    )
    
    st.header("📋 Informasi")
    st.info("""
    **Format gambar yang didukung:**
    - JPG, JPEG, PNG
    - Ukuran minimal: 128x128 pixel
    - Background polos lebih baik
    """)
    
    # Tampilkan info Python version
    st.code(f"Python: {np.__version__}")

# ============================================
# MAIN APP - SINGLE IMAGE PREDICTION
# ============================================
st.header("📷 Prediksi Single Image")

uploaded_file = st.file_uploader(
    "Pilih gambar tangan...",
    type=['jpg', 'jpeg', 'png'],
    key="single_upload"
)

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Gambar yang diupload", use_column_width=True)
    
    with col2:
        if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
            if model is None:
                st.error("Model tidak tersedia! Pastikan model file ada.")
            else:
                with st.spinner("Menganalisis gambar..."):
                    try:
                        # Predict
                        letter, confidence, all_probs = predict_image(image, model)
                        
                        # ============================================
                        # TAMPILKAN HASIL
                        # ============================================
                        st.subheader("🎯 Hasil Prediksi")
                        
                        # Huruf besar di tengah
                        st.markdown(f"""
                        <div style="text-align: center; padding: 30px; 
                        background-color: {'#d4edda' if confidence >= confidence_threshold else '#f8d7da'}; 
                        border-radius: 15px; margin: 20px 0; border: 3px solid {'#28a745' if confidence >= confidence_threshold else '#dc3545'}">
                            <h1 style="font-size: 96px; margin: 0; color: {'#155724' if confidence >= confidence_threshold else '#721c24'}">
                                {letter}
                            </h1>
                            <p style="font-size: 24px; margin: 10px 0;">Huruf Prediksi</p>
                            <p style="font-size: 20px; font-weight: bold; color: {'#155724' if confidence >= confidence_threshold else '#721c24'}">
                                Confidence: {confidence:.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Status prediksi
                        if confidence >= confidence_threshold:
                            st.success(f"✅ Prediksi YAKIN: Huruf '{letter}'")
                        else:
                            st.warning(f"⚠️ Confidence di bawah threshold ({confidence_threshold}%)")
                        
                        # Progress bar
                        st.progress(int(confidence) / 100, text=f"{confidence:.1f}%")
                        
                        # ============================================
                        # TOP 5 PREDICTIONS
                        # ============================================
                        st.subheader("🏆 Top 5 Predictions")
                        top_5_idx = np.argsort(all_probs)[-5:][::-1]
                        
                        for i, idx in enumerate(top_5_idx):
                            col_a, col_b, col_c = st.columns([1, 4, 2])
                            with col_a:
                                st.markdown(f"**{i+1}. {chr(65 + idx)}**")
                            with col_b:
                                prob = all_probs[idx] * 100
                                st.progress(
                                    float(prob/100), 
                                    text=f"{prob:.1f}%"
                                )
                            with col_c:
                                st.code(f"{prob:.2f}%")
                        
                        # ============================================
                        # ALL PREDICTIONS (expandable)
                        # ============================================
                        with st.expander("📊 Lihat Semua Prediksi (A-Z)"):
                            cols = st.columns(4)
                            for i in range(26):
                                with cols[i % 4]:
                                    prob = all_probs[i] * 100
                                    st.metric(
                                        label=chr(65 + i),
                                        value=f"{prob:.1f}%",
                                        delta="✓" if i == predicted_idx else None
                                    )
                        
                        # ============================================
                        # DOWNLOAD HASIL
                        # ============================================
                        st.subheader("💾 Download Hasil")
                        
                        # Buat file hasil
                        result_text = f"""HASIL PREDIKSI BISINDO
=========================
File: {uploaded_file.name}
Prediksi: {letter}
Confidence: {confidence:.2f}%
Timestamp: {st.session_state.get('timestamp', 'N/A')}

Detail Probabilitas:
"""
                        for i in range(26):
                            prob = all_probs[i] * 100
                            result_text += f"{chr(65 + i)}: {prob:.2f}%\n"
                        
                        # Tombol download
                        st.download_button(
                            label="📥 Download Hasil (TXT)",
                            data=result_text,
                            file_name=f"hasil_{uploaded_file.name.split('.')[0]}.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error saat prediksi: {str(e)}")
                        st.info("Pastikan gambar valid dan model compatible.")

# ============================================
# BATCH PREDICTION SECTION
# ============================================
st.header("📁 Batch Prediction")

uploaded_files = st.file_uploader(
    "Pilih multiple gambar...",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    key="batch_upload"
)

if uploaded_files and model:
    if st.button("🚀 Prediksi Semua", type="secondary", use_container_width=True):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Memproses {i+1}/{len(uploaded_files)}: {file.name}")
            
            try:
                image = Image.open(file)
                letter, confidence, _ = predict_image(image, model)
                
                results.append({
                    'File': file.name,
                    'Prediksi': letter,
                    'Confidence (%)': f"{confidence:.2f}",
                    'Status': '✓' if confidence >= confidence_threshold else '⚠️'
                })
                
            except Exception as e:
                results.append({
                    'File': file.name,
                    'Prediksi': 'ERROR',
                    'Confidence (%)': '0.00',
                    'Status': '❌',
                    'Error': str(e)
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ Selesai!")
        
        # Tampilkan tabel
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name="hasil_batch_bisindo.csv",
                mime="text/csv"
            )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>BISINDO Sign Language Classifier v1.0</p>
    <p>Deployed with ❤️ using Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# DEBUG INFO (hanya di development)
# ============================================
if st.sidebar.checkbox("Show debug info", False):
    st.sidebar.subheader("🔧 Debug Info")
    st.sidebar.write(f"Streamlit version: {st.__version__}")
    st.sidebar.write(f"NumPy version: {np.__version__}")
    st.sidebar.write(f"TensorFlow version: {keras.__version__}")
    st.sidebar.write(f"PIL version: {Image.__version__}")
    
    # Cek file di directory
    import subprocess
    result = subprocess.run(["ls", "-la"], capture_output=True, text=True)
    st.sidebar.code(result.stdout)