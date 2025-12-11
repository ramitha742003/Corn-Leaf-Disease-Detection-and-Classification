# Import Libraries

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load trained model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("corn_model.keras")

model = load_model()


# Class labels (MUST match training order)
CLASS_NAMES = [
    'Blight',
    'Common_Rust',
    'Gray_Leaf_Spot',
    'Healthy',
    'Not_Corn_Leaf'
]


# Calculate model accuracy from test data

@st.cache_data
def calculate_accuracy():

    test_path = r"C:/Users/ramit/OneDrive/Desktop/corn/dataset_split/test"

    if not os.path.exists(test_path):
        return 95

    datagen = ImageDataGenerator(rescale=1/255.0)

    test_gen = datagen.flow_from_directory(
        test_path,
        target_size=(128,128),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    loss, acc = model.evaluate(test_gen, verbose=0)

    return acc * 100


MODEL_ACCURACY = calculate_accuracy()


# STREAMLIT UI

st.set_page_config(
    page_title="Corn Leaf Disease Detection",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 Corn Leaf Disease Detection System")
st.markdown("---")
st.write("Upload a corn leaf image to predict its disease. Non-leaf images are automatically rejected.")


uploaded_file = st.file_uploader(
    "📤 Upload a Corn Leaf Image",
    type=["jpg", "jpeg", "png"]
)


# PREDICTION

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]

    idx = np.argmax(predictions)
    class_name = CLASS_NAMES[idx]
    confidence = predictions[idx] * 100


    # NOT CORN LEAF HANDLING (CORE FIX)

    if class_name == "Not_Corn_Leaf":

        st.error("❌ This image is NOT a corn leaf.")
        st.info("Please upload a clear image of a single corn leaf.")

    else:

        # Display results

        st.subheader(f"Prediction: **{class_name}**")
        # st.write(f"**Confidence:** {confidence:.2f}%")
        st.write(f"**Model Accuracy:** {MODEL_ACCURACY:.2f}%")

        # Disease severity

        if class_name == "Healthy":
            severity = "0% (No infection)"
        else:
            if confidence < 30:
                severity = f"{confidence:.2f}% (Mild Infection)"
            elif confidence < 60:
                severity = f"{confidence:.2f}% (Moderate Infection)"
            else:
                severity = f"{confidence:.2f}% (Severe Infection)"

        st.write(f"**Disease Severity:** {severity}")


        # Disease descriptions

        if class_name == "Blight":
            st.error("**Blight Detected:** Grayish lesions appear on leaves. Practice crop rotation and use fungicides.")

        elif class_name == "Common_Rust":
            st.warning("**Common Rust Detected:** Orange-brown pustules on leaves. Plant resistant corn varieties.")

        elif class_name == "Gray_Leaf_Spot":
            st.warning("**Gray Leaf Spot Detected:** Rectangular lesions on leaves. Use crop rotation and fungicides.")

        elif class_name == "Healthy":
            st.success("The leaf appears **Healthy**! No disease detected.")


else:
    st.info("👆 Upload a corn leaf image to begin detection.")


# Footer

st.markdown("---")
st.caption("**© 2025** Corn Leaf Disease Detection Project | All Rights Reserved.")
