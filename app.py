
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import cv2

# Load model
model = tf.keras.models.load_model('model.h5')

st.title("Facial Keypoints Detection")

uploaded_file = st.file_uploader("Upload a 96x96 grayscale face image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert('L').resize((96, 96))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(image).reshape(1, 96, 96, 1) / 255.0

    # Predict keypoints
    if st.button("Detect Keypoints"):
        keypoints = model.predict(img_array)[0].reshape(-1, 2)
        image_draw = image.convert('RGB')
        draw = ImageDraw.Draw(image_draw)
        for x, y in keypoints:
            draw.ellipse((x-1, y-1, x+1, y+1), fill=(255, 0, 0))
        st.image(image_draw, caption="Detected Facial Keypoints", use_column_width=True)
