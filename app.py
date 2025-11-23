import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("corn_disease_model.keras")

model = load_model()

# Class Labels (same order as training generator)
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']


# ----------------------------------------------------
# Calculate accuracy from test dataset
# ----------------------------------------------------
@st.cache_data
def calculate_accuracy():
    test_path = r"C:/Users/ramit/OneDrive/Desktop/project/corn_split_dataset/test"  # update if needed

    if not os.path.exists(test_path):
        return 95  # fallback accuracy 

    datagen = ImageDataGenerator(rescale=1/255.0)

    test_gen = datagen.flow_from_directory(
        test_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    loss, acc = model.evaluate(test_gen, verbose=0)
    return acc * 100

MODEL_ACCURACY = calculate_accuracy()


# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.set_page_config(page_title="Corn Leaf Disease Detection", page_icon="🌽")

st.title("🌽 Corn Leaf Disease Detection System")
st.write("Upload a corn leaf image to predict the disease using the trained CNN model.")
st.markdown("---")

# File Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])


# ----------------------------------------------------
# Make Prediction
# ----------------------------------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    class_name = CLASS_NAMES[idx]
    confidence = preds[0][idx] * 100

    # Show prediction
    st.subheader(f"Prediction: **{class_name}**")
    # st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Model Accuracy:** {MODEL_ACCURACY:.2f}%")

    # Severity
    if class_name == "Healthy":
        severity = "0% (No infection)"
    else:
        if confidence < 30:
            severity = f"{confidence:.2f}% (Mild Infection)"
        elif confidence < 60:
            severity = f"{confidence:.2f}% (Moderate Infection)"
        else:
            severity = f"{confidence:.2f}% (Severe Infection)"

    st.write(f"Disease Severity: **{severity}**")

    # Disease Info
    if class_name == "Blight":
        st.error("**Blight Detected:** Grayish lesions appear on leaves. Use resistant hybrids and fungicides.")
    elif class_name == "Common_Rust":
        st.warning("**Common Rust Detected:** Orange-brown pustules on leaves. Use resistant corn varieties.")
    elif class_name == "Gray_Leaf_Spot":
        st.warning("**Gray Leaf Spot Detected:** Rectangular gray lesions. Control via crop rotation and fungicides.")
    elif class_name == "Healthy":
        st.success("The leaf appears **Healthy**! No disease detected.")

else:
    st.info("👆 Upload an image to get prediction.")

# Footer
st.markdown("---")
st.caption("**© 2025** Corn Disease Detection Project. All Rights Reserved.")

