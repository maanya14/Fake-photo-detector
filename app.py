"""
Spot the Fake Photo — live camera demo (Streamlit + streamlit-webrtc)

Run with:
    streamlit run app.py

Point your camera at something real, then at a phone/laptop screen or a
printout, and watch the score update live.
"""

import av
import cv2
import joblib
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from feature_extractor import FeatureExtractor

# ---------------------------------------------------------------------------

st.set_page_config(page_title="Spot the Fake Photo — Live Demo", page_icon="📷", layout="centered")
st.title("📷 Spot the Fake Photo — Live Demo")
st.caption(
    "Point your camera at something real, then at a phone/laptop screen or a printout. "
    "Score runs 0 (real photo) → 1 (photo of a screen), live."
)


@st.cache_resource
def load_model():
    metadata = joblib.load("models/model.pkl")
    return metadata["model"], metadata["feature_length"]


model, feature_length = load_model()
extractor = FeatureExtractor()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold",
        0.0, 1.0, 0.38, 0.01,
        help="Score >= threshold is flagged as a screen recapture. "
             "0.38 is the value tuned via cross-validated probabilities in evaluate.py.",
    )
    process_every_n = st.slider(
        "Process every Nth frame",
        1, 10, 3,
        help="Feature extraction takes ~30-40ms. Skipping frames keeps the "
             "video feed smooth while still updating the score several times a second.",
    )
    st.markdown("---")
    st.markdown(
        "**Model:** SVM (RBF) · **Features:** ~56 (FFT/moiré, angular grid, "
        "sharpness, edges, LBP texture, color)\n\n"
        "**Honest CV accuracy:** 0.875 · **ROC AUC:** 0.934"
    )

# ---------------------------------------------------------------------------


class FakePhotoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_score = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % process_every_n == 0:
            try:
                feats = extractor.extract(img)
                if len(feats) == feature_length:
                    self.last_score = float(model.predict_proba(feats.reshape(1, -1))[0][1])
            except Exception:
                pass  # keep showing the last good score rather than crash the stream

        score = self.last_score
        is_screen = score >= threshold
        label = "SCREEN / RECAPTURE" if is_screen else "REAL"
        color = (0, 0, 255) if is_screen else (0, 200, 0)  # BGR

        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(img, f"{label}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(img, f"score = {score:.2f}  (threshold {threshold:.2f})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="spot-fake-photo",
    video_processor_factory=FakePhotoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.markdown(
    "Score near **0** → looks like a real photo. Score near **1** → looks like a "
    "photo of a screen (moiré/pixel-grid pattern, glare, flatter frequency spectrum). "
    "Everything runs locally in your browser tab + this Python process — no image "
    "is uploaded anywhere."
)
