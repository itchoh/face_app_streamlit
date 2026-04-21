import streamlit as st
import cv2
import face_recognition
import pickle

st.title("🎥 Face Recognition System")

# Load model
@st.cache_resource
def load_model():
    with open("model/test_save.clf", "rb") as f:
        return pickle.load(f)

model = load_model()

run = st.checkbox("Start Camera")

frame_window = st.image([])

# Camera state
if "camera" not in st.session_state:
    st.session_state.camera = None

if run:
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)

    cap = st.session_state.camera

    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
    else:
        # Resize
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        predictions = []
        if len(face_encodings) > 0:
            predictions = model.predict(face_encodings)

        # Draw results
        for (top, right, bottom, left), name in zip(face_locations, predictions):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        frame_window.image(frame, channels="BGR")

        # 🔥 This forces refresh (important)
        st.experimental_rerun()

else:
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None

    st.write("Camera stopped.")