# import streamlit as st
# import cv2
# import pickle
# import numpy as np
# import face_recognition

# st.set_page_config(page_title="Face Recognition", layout="centered")
# st.title("🎥 Face Recognition System")

# # ---------------------------
# # Load model
# # ---------------------------
# @st.cache_resource
# def load_model():
#     with open("test_save.clf", "rb") as f:
#         return pickle.load(f)

# model = load_model()

# st.write("Model loaded:", model)

# # ---------------------------
# # Camera
# # ---------------------------
# img_file = st.camera_input("Take a photo")

# # ---------------------------
# # Process
# # ---------------------------
# if img_file is not None and model is not None:

#     file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # detect faces
#     face_locations = face_recognition.face_locations(rgb)

#     if len(face_locations) == 0:
#         st.warning("No face detected")
#         st.image(img, channels="BGR")
#         st.stop()

#     for (top, right, bottom, left) in face_locations:

#         # ❗ SAME AS TRAINING: encoding (NOT pixels)
#         encoding = face_recognition.face_encodings(
#             rgb,
#             [(top, right, bottom, left)]
#         )

#         name = "Unknown"

#         if len(encoding) > 0:
#             try:
#                 encoding = encoding[0].reshape(1, -1)
#                 name = model.predict(encoding)[0]
#             except Exception as e:
#                 st.error(f"Prediction error: {e}")

#         cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(img, str(name), (left, top - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     st.image(img, channels="BGR")
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
    with open("test_save.clf", "rb") as f:
        return pickle.load(f)

model = load_model()

st.write("Model loaded successfully")

# ---------------------------
# Face detector
# ---------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

img_file = st.camera_input("Take a photo")

# ---------------------------
# Process
# ---------------------------
if img_file is not None and model is not None:

    file_bytes = np.asarray(bytearray(img_file.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        st.warning("No face detected")
        st.image(img, channels="BGR")
        st.stop()

    for (x, y, w, h) in faces:

        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        face = face.flatten().reshape(1, -1)

        try:
            name = model.predict(face)[0]
        except:
            name = "Unknown"

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(name), (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    st.image(img, channels="BGR")
