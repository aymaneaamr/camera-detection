import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title="Compteur Simple", page_icon="🔢")

st.title("📸 Compteur d'Objets")
st.write("Application de démonstration - Compatible Python 3.13")

option = st.radio("Source :", ["Caméra", "Upload image", "Test"])

if option == "Caméra":
    img_file = st.camera_input("Prenez une photo")
    if img_file is not None:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Détection de contours
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dessin
        cv2.drawContours(cv2_img, contours, -1, (0, 255, 0), 2)
        cv2.putText(cv2_img, f"Objets: {len(contours)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        st.image(cv2_img, channels="BGR")

elif option == "Upload image":
    uploaded = st.file_uploader("Choisir image", type=['jpg', 'png'])
    if uploaded:
        image = Image.open(uploaded)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            result = img_array.copy()
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            cv2.putText(result, f"Objets: {len(contours)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            st.image(result)

else:  # Test
    st.info("Test de détection sur image générée")
    test = np.zeros((400, 600, 3), dtype=np.uint8)
    test.fill(255)
    cv2.circle(test, (200, 200), 50, (100, 100, 100), -1)
    cv2.circle(test, (400, 200), 40, (150, 150, 150), -1)
    
    gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(test, contours, -1, (0, 255, 0), 2)
    cv2.putText(test, f"Objets: {len(contours)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    st.image(test)
