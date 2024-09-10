import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
import sqlite3

# Database connection
conn = sqlite3.connect('user_data.db')
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS users (
             username TEXT PRIMARY KEY,
             password TEXT,
             email TEXT,
             phone TEXT,
             address TEXT
             )''')
conn.commit()

# Define image sizes for each model
image_sizes = {
    "Mobilenet": (224, 224, 3),
    "Resnet": (224, 224, 3),
    "Sequential": (256, 256, 3)
}

# Define paths for three different models
model_paths = {
    "Sequential": r"C:\Users\ravin\MAJOR_PROJECT\model_sequential.h5",
    "Mobilenet": r"C:\Users\ravin\MAJOR_PROJECT\MobileNetV2.h5",
    "Resnet": r"C:\Users\ravin\MAJOR_PROJECT\Resnet50_with_Attention.h5"
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

# Streamlit UI
st.title('Remote sensing Scene Classification')

# Login and Signup
page = st.sidebar.radio("Action", ("Login", "Signup"))

# if page == "Login":
#     st.header("Login")
#     with st.form(key="login_form"):
#         login_username = st.text_input("Username")
#         login_password = st.text_input("Password", type="password")
#         submit_login = st.form_submit_button("Login")
#         if submit_login:
#             # Check login credentials
#             c.execute("SELECT * FROM users WHERE username=? AND password=?", (login_username, login_password))
#             if c.fetchone():
#                 st.success("Login Successful")
#                 # File upload section
#                 st.subheader("Upload Image")
#                 uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

#                 # Dropdown menu for selecting model
#                 selected_model = st.selectbox("Select Model", list(model_paths.keys()))

#                 if uploaded_file is not None:
#                     st.image(uploaded_file, caption='Uploaded Image')

#                 if st.button('Predict'):
#                     # Load the selected model
#                     model = load_selected_model(selected_model)
#                     image_size = image_sizes[selected_model]
#                     prediction = predict(uploaded_file, model, image_size)
#                     st.write(prediction)
#             else:
#                 st.error("Invalid username or password")

# elif page == "Signup":
#     st.header("Signup")
#     with st.form(key="signup_form"):
#         signup_username = st.text_input("New Username")
#         signup_password = st.text_input("New Password", type="password")
#         signup_email = st.text_input("Email")
#         signup_phone = st.text_input("Phone")
#         signup_address = st.text_input("Address")
#         submit_signup = st.form_submit_button("Signup")
#         if submit_signup:
#             # Add new user to the database
#             c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
#                       (signup_username, signup_password, signup_email, signup_phone, signup_address))
#             conn.commit()
#             st.success("Signup Successful. Please login now.")
