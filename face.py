import streamlit as st
import cv2
import pickle
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ✅ NEW MediaPipe API
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

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
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---------------------------
# Load MediaPipe Face Detector
# ---------------------------
@st.cache_resource
def load_face_detector():
    try:
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path="face_detector.tflite"),
            min_detection_confidence=0.5
        )
        return vision.FaceDetector.create_from_options(options)
    except Exception as e:
        st.error(f"Error loading face detector: {e}")
        return None

face_detector = load_face_detector()

# ---------------------------
# Transformer
# ---------------------------
class FaceRecTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if img is None or face_detector is None:
            return img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = vision.Image(
            image_format=vision.ImageFormat.SRGB,
            data=rgb
        )

        results = face_detector.detect(mp_image)

        if results and results.detections:
            h, w, _ = img.shape

            for detection in results.detections:
                bbox = detection.bounding_box

                x = max(0, int(bbox.origin_x))
                y = max(0, int(bbox.origin_y))
                x2 = min(w, x + int(bbox.width))
                y2 = min(h, y + int(bbox.height))

                face_img = img[y:y2, x:x2]

                name = "Unknown"

                if model is not None and face_img.size > 0:
                    try:
                        face_resized = cv2.resize(face_img, (128, 128))
                        face_flat = face_resized.flatten().reshape(1, -1)
                        name = model.predict(face_flat)[0]
                    except:
                        pass

                cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    str(name),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        return img


# ---------------------------
# WebRTC Stream (FIXED)
# ---------------------------
webrtc_streamer(
    key="face-recognition",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=FaceRecTransformer,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
