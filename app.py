import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
import cv2

def crop_to_content(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mask = gray < 200
    coords = np.argwhere(mask)

    if coords.size == 0:
        return image

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1

    cropped = image[x_min:x_max, y_min:y_max]
    return cropped

@st.cache_resource
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

st.title("Pengenalan Aksara Sunda")

option = st.selectbox(
    'Pilih model:',
    ('model1.keras', 'model2.keras', 'model3.keras', 'model4.keras'),
)

model = load_model(option)

col1, col2 = st.columns(2, vertical_alignment='center')

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Warna kuas
        stroke_width=5,                      # Ketebalan kuas
        stroke_color="#000000",              # Warna garis
        background_color="#FFFFFF",         # Warna background
        height=200,                           # Tinggi canvas
        width=200,                            # Lebar canvas
        drawing_mode="freedraw",             # Mode menggambar
        key="canvas",
    )

def predict_image(canvas_data):
    if canvas_data is not None:
        img_array = canvas_data[:, :, :3]
        
        cropped_img = crop_to_content(img_array)
        
        resized_img = cv2.resize(cropped_img, (64, 64))
        
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img / 255.0
        gray_img = gray_img.reshape(1, 64, 64, 1)
        
        if model:
            prediction = model.predict(gray_img)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction)
            return predicted_label, confidence
        else:
            return None, None
    return None, None

with col2:
    if canvas_result.image_data is not None:
        predicted_label, confidence = predict_image(canvas_result.image_data)
        class_labels = ['a','ae','ba','ca','da','e','eu','fa','ga','ha','i','ja','ka','la','ma','na','nga','nya','o','pa','qa','ra','sa','ta','u','va','wa','xa','ya','za']

        if predicted_label is not None:
            st.write(f"### Prediksi: {class_labels[predicted_label]}")
            st.write(f"### Keyakinan: {confidence:.2f}")
        else:
            st.write("### Ceker Hayam")