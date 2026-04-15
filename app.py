import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
model = YOLO('best.pt')

st.title("Face Mask Detection Model Using YOLOv8")
st.sidebar.title("Select an Option")
option = st.sidebar.selectbox("Choose Mode", ("Upload Image", "Live Webcam Detection"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        results = model(img_np,conf=0.5)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detection Result", width=600)

elif option == "Live Webcam Detection":
    stframe = st.empty()
    run = st.checkbox('Start Webcam')
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not accessible")
                break
            results = model(frame)
            res_plotted = results[0].plot()
            stframe.image(res_plotted, channels="BGR")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()