import streamlit as st
import cv2
import face_recognition
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("🎥 Face Recognition System")

@st.cache_resource
def load_model():
    with open("model/test_save.clf", "rb") as f:
        return pickle.load(f)

model = load_model()

class FaceRecTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locs)

        names = []
        if len(encodings) > 0:
            names = model.predict(encodings)

        for (top, right, bottom, left), name in zip(locs, names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(img, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        return img

webrtc_streamer(key="face-recognition", video_transformer_factory=FaceRecTransformer)
