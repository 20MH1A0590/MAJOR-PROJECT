import streamlit as st
import numpy as np
from numpy import asarray
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Define image sizes for each model
image_sizes = {
    "Mobilenet": (224, 224, 3),
    "Resnet": (224, 224, 3),
    "Sequential": (256, 256, 3)
}

batch_size = 32

# Define paths for three different models
model_paths = {
    "Sequential": r"C:\Users\ravin\MAJOR_PROJECT\model_sequential.h5",
    ##"Mobilenet": r"C:\Users\ravin\MAJOR_PROJECT\MobileNetV2.h5",
    ##"Resnet": r"C:\Users\ravin\MAJOR_PROJECT\Resnet50_with_Attention.h5"
}
# Load model based on user's choice
def load_selected_model(model_name):
    return load_model(model_paths[model_name])

# Function to make prediction
def predict(image_path, model, image_size):
    class_labels = {
        'aGrass': 0,
        'bField': 1,
        'cIndustry': 2,
        'dRiverLake': 3,
        'eForest': 4,
        'fResident': 5,
        'gParking': 6}
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = list(class_labels.keys())[predicted_class_index]

    return predicted_class_label


        
st.title('Remote sensing Scene Classification')

# File upload section
imge = st.file_uploader('Upload your file', type=['JPG', 'PNG', 'JPEG', 'TIFF'], accept_multiple_files=False, key=None, help=None,
                        on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

# Dropdown menu for selecting model
selected_model = st.selectbox("Select Model", list(model_paths.keys()))

if (imge is not None):
    st.image(imge, caption='Uploaded Image')

if st.button('Predict'):
    # Load the selected model
    model = load_selected_model(selected_model)
    image_size = image_sizes[selected_model]
    prediction = predict(imge, model, image_size)
    st.markdown(""" <style> .predict {
font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)
    st.write(prediction)