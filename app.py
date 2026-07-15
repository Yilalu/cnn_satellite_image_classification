import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "saved_models/image_model_v2.keras"
    )
    return model


model = load_model()


# Class names (make sure this matches your training class_names)
classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


st.title("🛰️ Satellite Image Recognition")

st.write(
    "Upload a satellite image and the model will classify it."
)


# Debug model information
st.write("Model input shape:", model.input_shape)
st.write("Model output shape:", model.output_shape)


# Upload image
uploaded_file = st.file_uploader(
    "Choose a satellite image",
    type=["jpg", "jpeg", "png", "tif"]
)


if uploaded_file:

    # Open image
    image = Image.open(uploaded_file)

    # Make sure image has 3 channels
    image = image.convert("RGB")


    st.image(
        image,
        caption="Uploaded Image",
        width="content"
    )


    # -------------------------
    # Preprocessing
    # -------------------------

    # Resize to model expected size
    img = image.resize((128, 128))


    # Convert to numpy array
    img_array = np.array(img)


    # Convert to float32
    # DO NOT divide by 255
    # Model already contains Rescaling(1./255)
    img_array = img_array.astype("float32")


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # Debug
    st.write(
        "Input shape:",
        img_array.shape
    )

    st.write(
        "Pixel range:",
        img_array.min(),
        "-",
        img_array.max()
    )


    # -------------------------
    # Prediction
    # -------------------------

    prediction = model.predict(img_array)


    st.write(
        "Raw prediction:",
        prediction
    )


    class_id = np.argmax(prediction[0])

    confidence = np.max(prediction[0])


    st.subheader("Prediction")


    st.success(
        classes[class_id]
    )


    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )


    # -------------------------
    # All class probabilities
    # -------------------------

    st.subheader(
        "Class Probabilities"
    )


    for i, prob in enumerate(prediction[0]):

        st.write(
            f"{classes[i]}: {prob * 100:.2f}%"
        )