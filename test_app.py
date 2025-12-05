# test_app.py - Untuk testing deployment
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="BISINDO Test", layout="centered")

st.title("🎯 BISINDO Deployment Test")
st.markdown("Testing basic functionality without TensorFlow")

# Test 1: Basic imports
st.subheader("✅ Import Test")
st.code("""
import streamlit ✓
import numpy ✓  
import PIL ✓
""")

# Test 2: File upload
st.subheader("📁 File Upload Test")
uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    
    with col2:
        # Resize test
        resized = image.resize((128, 128))
        st.image(resized, caption="Resized 128x128", use_column_width=True)
    
    # Numpy test
    img_array = np.array(image)
    st.write(f"**Image Info:** Shape={img_array.shape}, Max={img_array.max()}, Min={img_array.min()}")

# Test 3: Model file check
st.subheader("🔍 Model File Check")
import os

if os.path.exists('final_bisindo_model.keras'):
    st.success("✅ Model file found: final_bisindo_model.keras")
    file_size = os.path.getsize('final_bisindo_model.keras') / (1024 * 1024)
    st.write(f"File size: {file_size:.2f} MB")
else:
    st.error("❌ Model file NOT found!")
    st.info("Upload 'final_bisindo_model.keras' to your GitHub repository")

st.markdown("---")
st.info("Jika semua test berhasil, tambahkan TensorFlow untuk prediksi.")