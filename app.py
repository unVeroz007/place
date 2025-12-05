# app.py - ULTRA SIMPLE VERSION
import streamlit as st
import numpy as np
from PIL import Image
import os

st.title("🤟 BISINDO Classifier - Testing")
st.write("Testing deployment tanpa TensorFlow dulu")

# Check files in directory
st.subheader("📁 Files in directory:")
try:
    files = os.listdir('.')
    st.write(f"Found {len(files)} files:")
    for file in files:
        st.write(f"- {file}")
except Exception as e:
    st.error(f"Error listing files: {e}")

# Simple file upload test
uploaded_file = st.file_uploader("Upload test image", type=['jpg', 'png'])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        st.success("✅ Image loaded successfully!")
        
        # Convert to numpy
        img_array = np.array(image)
        st.write(f"Image shape: {img_array.shape}")
        st.write(f"Image dtype: {img_array.dtype}")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")