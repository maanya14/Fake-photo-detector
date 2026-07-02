
import cv2
import joblib
import numpy as np
import streamlit as st

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


st.subheader("Capture an Image")

image = st.camera_input("Take a picture")

if image is not None:
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    try:
        features = extractor.extract(img)

        if len(features) != feature_length:
            st.error(
                f"Feature mismatch: expected {feature_length}, got {len(features)}"
            )
        else:
            score = float(
                model.predict_proba(features.reshape(1, -1))[0][1]
            )

            is_screen = score >= threshold

            if is_screen:
                st.error("SCREEN / RECAPTURE DETECTED")
            else:
                st.success("REAL PHOTO")

            st.metric("Detection Score", f"{score:.3f}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Captured Image", use_container_width=True)

    except Exception as e:
        st.exception(e)
        
st.markdown("---")
st.markdown(
    "Score near **0** → looks like a real photo. Score near **1** → looks like a "
    "photo of a screen (moiré/pixel-grid pattern, glare, flatter frequency spectrum). "
    "Everything runs locally in your browser tab + this Python process — no image "
    "is uploaded anywhere."
)
