import streamlit as st
import cv2
import pickle
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("🎥 Face Recognition System")

# Load model
@st.cache_resource
def load_model():
    try:
        with open("test_save.clf", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ✅ MediaPipe (compatible with mediapipe==0.9.3.0)
mp_face = mp.solutions.face_detection

class FaceRecTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_detection = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if img is None:
            return img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if results and results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape

                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Prevent out-of-bounds crop
                x2 = min(w, x + width)
                y2 = min(h, y + height)

                face_img = img[y:y2, x:x2]

                name = "Unknown"

                if model is not None and face_img.size > 0:
                    try:
                        face_resized = cv2.resize(face_img, (128, 128))
                        face_flat = face_resized.flatten().reshape(1, -1)
                        name = model.predict(face_flat)[0]
                    except Exception:
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


webrtc_streamer(
    key="face-recognition",
    video_transformer_factory=FaceRecTransformer
)
