import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


@st.cache_resource
def load_skin_model():
    return tf.keras.models.load_model("models/skin_non_skin_model.h5")

@st.cache_resource
def load_eczema_model():
    return tf.keras.models.load_model("models/eczema_classification_model-1(acc_89.03,loss_0.2956).h5")

skin_model = load_skin_model()
eczema_model = load_eczema_model()

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image


st.title("Skin Eczema Classification with TensorFlow")

verify_skin = st.checkbox("Verify Skin", True)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    label = "Skin"
    processed_image = preprocess_image(image)
    if verify_skin:
        st.write("Verifying Skin...")
        predictions = skin_model.predict(processed_image)

        confidence = predictions[0][0]
        if confidence > 0.5:
            label = "Skin"
        else:
            label = "Non-Skin"

    if label == "Skin":
        eczema_prediction = eczema_model.predict(processed_image)
        eczema_labels = ["Atopic Eczema", "Contact Dermatitis", "Nummular Eczema"]
        predicted_label = eczema_labels[np.argmax(eczema_prediction)]

        st.markdown(f"### Prediction: **{predicted_label}**")
        st.markdown(f"### Confidence: **{np.max(eczema_prediction):.2f}**")
    else:
        st.markdown("### Image is not a skin image")