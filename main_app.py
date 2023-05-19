import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = tf.keras.models.load_model(final_model_best.h5)
    return model

def predict_image_class(image_path, model_path):
    # Load the trained model
    model = load_model(final_model_best.h5)

    # Load the image and preprocess it
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img)
    img_array = img_array.reshape((1, 28, 28, 1))
    img_array = img_array.astype('float32')
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name

st.write("# Clothes Detection System by Iñigo Teope")

file = st.file_uploader("Choose clothes photo from computer", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class_name = predict_image_class(file, 'final_model_best.h5')
    st.success("OUTPUT: " + predicted_class_name)
