import streamlit as st
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="kux79LtKQfMKceSYUNtO"
)

st.set_page_config(page_title="Arabic Sign Language Detector", page_icon="khaled", layout="centered")
st.title("Arabic Sign Language Detection App")
st.write("Upload an image to detect Arabic Sign Language gestures using Roboflow AI.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Run inference
    with st.spinner("Running detection..."):
        result = CLIENT.infer(tmp_path, model_id="arabic-sign-language-translator-tvlbp/2")

    # Process image
    img = cv2.imread(tmp_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw predictions
    detections = result.get("predictions", [])
    if detections:
        for pred in detections:
            x, y = int(pred["x"]), int(pred["y"])
            w, h = int(pred["width"]), int(pred["height"])
            cls = pred["class"]
            conf = pred["confidence"]

            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show the detected image
        st.image(img, caption=" Detected Arabic Signs", use_container_width=True)

        # Show the classes
        st.subheader("Detected Signs:")
        for pred in detections:
            st.write(f"- **{pred['class']}** (Confidence: {pred['confidence']:.2f})")

    else:
        st.warning("No signs detected. Try another image!")

    st.success(" Detection complete!")

st.markdown("---")
st.caption("Built with using Streamlit a.")
