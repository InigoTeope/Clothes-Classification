import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('final_model_best.h5')
    return model

model = load_model()

st.write("# Clothes Classification System by Iñigo")

file = st.file_uploader("Choose clothes photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (28, 28)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = ImageOps.grayscale(image)
    img = np.asarray(image)
    img = img.reshape((size[0], size[1], 1))  # Reshape to (28, 28, 1)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    string = "OUTPUT: " + class_names[np.argmax(prediction)]
    st.success(string)
