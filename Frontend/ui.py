import streamlit as st
import requests
import base64

# FastAPI backend URL
API_URL = "http://127.0.0.1:8001/predict/"

st.title("Resnet50 Classifier")


# Model selection
model_key = st.sidebar.selectbox(
    "Select model",
    ["bike_car_model"]  # Add more if your backend supports
)

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.markdown("## Please Upload an image !!! ")

if uploaded_file is not None:
    # Show uploaded image preview
    st.image(uploaded_file, caption="Preview of uploaded image",
             use_column_width=True)

    # Read file as bytes and encode to base64
    image_bytes = uploaded_file.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    if st.sidebar.button("Predict"):
        with st.spinner("In Progress..."):
            # import time
            # time.sleep(3)
            # Prepare payload
            payload = {
                "model": model_key,
                "encoded_image": encoded_image
            }

            try:
                # Send POST request
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    st.sidebar.success(
                        f"✅ Prediction: **{result['predicted_class']}**")
                    st.sidebar.info(f"Confidence: {result['confidence']}")
                else:
                    st.sidebar.error(
                        f"❌ Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"⚠️ Request failed: {e}")

# Add footer
st.markdown(
    """
    <hr style="margin-top: 2rem; margin-bottom: 1rem;">
    <div style='text-align: center; color: grey;'>
        Made by <b>SonakshiShingade 😸🌻</b>
    </div>
    """,
    unsafe_allow_html=True
)
