import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import sys, streamlit as st
st.write("Using Python from:", sys.executable)

@st.cache_resource
def load_emotion_model():
    model_path = r"E:\waste-classifier\face_emotion\emotion_model_final.h5"
    return load_model(model_path)

model = load_emotion_model()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


st.title("Face Emotion Detection")
st.write("Upload an image to detect the emotion.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    img = np.array(image)
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

   
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected in the image!")
    else:
        x, y, w, h = faces[0]  
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48,48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)

        pred = model.predict(face_img)[0]
        predicted_emotion = emotion_labels[np.argmax(pred)]
        st.success(f"Predicted Emotion: {predicted_emotion}")

     
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(emotion_labels, pred, color='skyblue')
        ax.set_ylabel("Probability")
        ax.set_title("Emotion Prediction Probabilities")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        
        
