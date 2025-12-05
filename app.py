# app.py - FINAL WORKING VERSION
import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="BISINDO Classifier",
    page_icon="✋",
    layout="centered"
)

st.title("🤟 BISINDO Sign Language Classifier")
st.markdown("Upload gambar tangan untuk mengenali huruf BISINDO (A-Z)")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.info("""
    **Cara penggunaan:**
    1. Upload gambar tangan
    2. Klik tombol Prediksi
    3. Lihat hasil prediksi
    
    **Format yang didukung:**
    - JPG, JPEG, PNG
    - Background polos lebih baik
    """)

# Function to check if TensorFlow is available
def check_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow import keras
        return True, tf, keras
    except ImportError as e:
        return False, None, None

# Check TensorFlow
tf_available, tf, keras = check_tensorflow()

if not tf_available:
    st.warning("⚠️ TensorFlow belum diinstall. Menginstall...")
    
    # Try to install TensorFlow
    import subprocess
    import sys
    
    try:
        # Install TensorFlow
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0"])
        st.success("✅ TensorFlow berhasil diinstall!")
        
        # Reload
        st.rerun()
    except Exception as e:
        st.error(f"❌ Gagal menginstall TensorFlow: {e}")
        st.info("""
        Jika error terus muncul, coba:
        1. Pastikan requirements.txt berisi: tensorflow==2.13.0
        2. Hapus cache di Streamlit Cloud
        """)
else:
    st.success("✅ TensorFlow tersedia!")
    
    # Load model
    @st.cache_resource
    def load_model():
        try:
            # Cek beberapa lokasi yang mungkin
            model_paths = [
                'final_bisindo_model.keras',
                './final_bisindo_model.keras',
                '/mount/src/place/final_bisindo_model.keras'
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    model = keras.models.load_model(path)
                    st.sidebar.success(f"✅ Model loaded: {path}")
                    return model
            
            st.sidebar.error("❌ Model file tidak ditemukan!")
            st.sidebar.info("""
            Pastikan file 'final_bisindo_model.keras' ada di repository GitHub.
            """)
            return None
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)[:100]}")
            return None
    
    model = load_model()
    
    # Image preprocessing function (without OpenCV)
    def preprocess_image(image, target_size=(128, 128)):
        """Preprocess image using PIL only"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image_resized = image.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(image_resized)
        img_normalized = img_array / 255.0
        
        return img_normalized
    
    # Main app
    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar tangan yang menunjukkan huruf BISINDO"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Gambar yang diupload", width=250)
        
        with col2:
            if st.button("🔍 Prediksi", type="primary", use_container_width=True):
                if model is None:
                    st.error("Model tidak tersedia. Pastikan model file ada.")
                else:
                    with st.spinner("Menganalisis gambar..."):
                        try:
                            # Preprocess and predict
                            img_processed = preprocess_image(image)
                            img_input = np.expand_dims(img_processed, axis=0)
                            
                            predictions = model.predict(img_input, verbose=0)
                            predicted_idx = np.argmax(predictions[0])
                            confidence = predictions[0][predicted_idx] * 100
                            letter = chr(65 + predicted_idx)  # 0->A, 1->B, etc.
                            
                            # Display results
                            st.subheader("🎯 Hasil Prediksi")
                            
                            # Big letter display
                            st.markdown(f"""
                            <div style="text-align: center; padding: 30px; 
                            background-color: #4CAF50; color: white; 
                            border-radius: 15px; margin: 20px 0;">
                                <h1 style="font-size: 96px; margin: 0;">{letter}</h1>
                                <p style="font-size: 24px;">Huruf Prediksi</p>
                                <p style="font-size: 20px;">Confidence: {confidence:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence indicator
                            if confidence >= 90:
                                st.success("✅ Prediksi sangat yakin!")
                            elif confidence >= 70:
                                st.warning("⚠️ Prediksi cukup yakin")
                            else:
                                st.error("❌ Confidence rendah, hasil mungkin tidak akurat")
                            
                            # Progress bar
                            st.progress(int(confidence) / 100)
                            
                            # Show top 3 predictions
                            st.subheader("🏆 Top 3 Predictions")
                            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                            
                            for i, idx in enumerate(top_3_idx):
                                col_a, col_b = st.columns([1, 4])
                                with col_a:
                                    st.markdown(f"**{i+1}. {chr(65 + idx)}**")
                                with col_b:
                                    prob = predictions[0][idx] * 100
                                    st.progress(float(prob/100), text=f"{prob:.1f}%")
                            
                            # Download result
                            result_text = f"""HASIL PREDIKSI BISINDO
=========================
File: {uploaded_file.name}
Prediksi: {letter}
Confidence: {confidence:.2f}%

Detail:
"""
                            for i in range(26):
                                prob = predictions[0][i] * 100
                                result_text += f"{chr(65 + i)}: {prob:.2f}%\n"
                            
                            st.download_button(
                                label="📥 Download Hasil",
                                data=result_text,
                                file_name=f"hasil_{uploaded_file.name.split('.')[0]}.txt",
                                mime="text/plain"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error saat prediksi: {str(e)}")
    
    # Debug section
    if st.sidebar.checkbox("Show debug info", False):
        st.sidebar.subheader("🔧 Debug Info")
        st.sidebar.write(f"TensorFlow version: {tf.__version__}")
        st.sidebar.write(f"NumPy version: {np.__version__}")
        
        # List files
        try:
            files = os.listdir('.')
            st.sidebar.write(f"Files in directory ({len(files)}):")
            for file in files[:10]:  # Show first 10 files
                st.sidebar.write(f"  - {file}")
        except:
            pass

# Footer
st.markdown("---")
st.caption("BISINDO Sign Language Classifier | Deployed on Streamlit Cloud")