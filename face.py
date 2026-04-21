import streamlit as st
import cv2
import pickle
import numpy as np
import face_recognition

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("🎥 Face Recognition System")

# ---------------------------
# Load model (KNN on embeddings)
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
# Camera
# ---------------------------
st.subheader("📸 Capture Image")
img_file = st.camera_input("Take a photo")

# ---------------------------
# Process
# ---------------------------
if img_file is not None and model is not None:

    file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------------
    # FACE DETECTION (face_recognition)
    # ---------------------------
    face_locations = face_recognition.face_locations(rgb)

    if len(face_locations) == 0:
        st.warning("No face detected")
        st.image(img, channels="BGR")
        st.stop()

    # ---------------------------
    # PREDICTION
    # ---------------------------
    for (top, right, bottom, left) in face_locations:

        face_encodings = face_recognition.face_encodings(
            rgb,
            [(top, right, bottom, left)]
        )

        name = "Unknown"

        if len(face_encodings) > 0:
            encoding = face_encodings[0].reshape(1, -1)

            name = model.predict(encoding)[0]

        # draw box
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            img,
            str(name),
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    st.image(img, channels="BGR")
