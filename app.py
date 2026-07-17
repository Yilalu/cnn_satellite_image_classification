import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image


# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "saved_models/image_model_v2.keras"
    )
    return model


model = load_model()


# Class names
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


st.set_page_config(page_title="Satellite Image Recognition", layout="centered")

st.title("Satellite Image Recognition")
st.write("Upload a satellite image and the model will classify it into one of 10 land-cover types.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a satellite image",
    type=["jpg", "jpeg", "png", "tif"]
)


if uploaded_file:

    # Open image
    image = Image.open(uploaded_file)
    image = image.convert("RGB")

    # -------------------------
    # Preprocessing
    # -------------------------
    img = image.resize((64, 64))
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # -------------------------
    # Prediction
    # -------------------------
    prediction = model.predict(img_array, verbose=0)[0]
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_class = classes[class_id]

    # -------------------------
    # Layout: image on the left, headline result on the right
    # -------------------------
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Prediction")
        st.success(f"**{predicted_class}**")
        st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")

        # Simple visual confidence indicator
        st.progress(float(confidence))

        if confidence < 0.5:
            st.warning("⚠️ Low confidence — the model isn't very sure about this one.")

    # -------------------------
    # Class probability chart
    # -------------------------
    st.subheader("Class Probabilities")

    prob_df = pd.DataFrame({
        "Class": classes,
        "Probability": prediction * 100
    }).sort_values("Probability", ascending=True)

    # Highlight the predicted class differently from the rest
    prob_df["Predicted"] = prob_df["Class"] == predicted_class

    st.bar_chart(
        prob_df.set_index("Class")["Probability"],
        horizontal=True,
    )

    # exact numbers in a compact table below the chart
    with st.expander("See exact probabilities"):
        st.dataframe(
            prob_df[["Class", "Probability"]]
            .sort_values("Probability", ascending=False)
            .style.format({"Probability": "{:.2f}%"}),
            hide_index=True,
            use_container_width=True,
        )

    # -------------------------
    # Debug info tucked away instead of cluttering the main view
    # -------------------------
    with st.expander("Detailed info"):
        st.write("Model input shape:", model.input_shape)
        st.write("Model output shape:", model.output_shape)
        st.write("Preprocessed input shape:", img_array.shape)
        st.write("Pixel range:", img_array.min(), "-", img_array.max())
        st.write("Raw prediction vector:", prediction)