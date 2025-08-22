import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json


# Load model once and cache it
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/chest_anomaly_model.h5")
    return model

# Predict disease
def predict_disease(img, model, class_names):
    x = image.img_to_array(img)        # convert image to array
    x = x / 255.0                      # normalize pixel values (0–1)
    x = np.expand_dims(x, axis=0)      # add batch dimension
    preds = model.predict(x, verbose=0)
    predicted_index = np.argmax(preds) # index of highest probability
    disease_name = class_names[predicted_index]
    accuracy = float(preds[0][predicted_index])
    return disease_name, accuracy


# Main function
def main():
    st.title (" chest x_ray Anomaly detection App")

    # Sidebar with mode selection
    st.sidebar.title("Options")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Choose Mode:",
        ["🔍 Image Prediction", "🤖 Chatbot"],
        help="Select the mode you want to use"
    )

 # Main content based on selected mode
    
    if mode == "🔍 Image Prediction":
        st.subheader("📸 chest Anomaly  detection")
        uploaded_file = st.file_uploader("📤 Upload a X ray image...", type=["jpg", "jpeg", "png"])

        model = load_model()
        
        if model is None:
            st.error("❌ Failed to load model. Please check if the model file exists.")
            return

        # Load class names from JSON
        try:
            with open("class_names.json", "r") as f:
                class_names = json.load(f)
        except Exception as e:
            st.error(f"❌ Error loading class names: {str(e)}")
            return

        if uploaded_file is not None:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            st.image(img, caption="Uploaded x-ray Image", use_container_width=True)

            if st.button("🔍 Predict Disease"):
                disease_name, accuracy = predict_disease(img, model, class_names)
                
                st.success(f"Disease Detected: **{disease_name}**")
                st.info(f"Model Accuracy: **{accuracy * 100:.2f}%**")
                

    elif mode == "🤖 Chatbot":
        pass

if __name__ == "__main__":
    main()
        
        