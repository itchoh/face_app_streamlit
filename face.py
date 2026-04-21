import streamlit as st
import cv2
import pickle
import numpy as np

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("🎥 Face Recognition System")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    try:
        with open("test_save.clf", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

model = load_model()

# ---------------------------
# OpenCV Face Detector (STABLE)
# ---------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.subheader("📸 Capture Image")

img_file = st.camera_input("Take a photo")

# ---------------------------
# Process Image
# ---------------------------
if img_file is not None:

    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        face_img = img[y:y+h, x:x+w]

        name = "Unknown"

        if model is not None and face_img.size > 0:
            try:
                face_resized = cv2.resize(face_img, (128, 128))
                face_flat = face_resized.flatten().reshape(1, -1)
                name = model.predict(face_flat)[0]
            except:
                pass

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            img,
            str(name),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    st.image(img, channels="BGR")
