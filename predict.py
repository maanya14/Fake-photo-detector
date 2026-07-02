import sys
import cv2
import joblib
import numpy as np

from feature_extractor import FeatureExtractor

MODEL_PATH = "models/model.pkl"

# -----------------------
# Load model
# -----------------------

metadata = joblib.load(MODEL_PATH)
model = metadata["model"]
feature_length = metadata["feature_length"]
extractor = FeatureExtractor()

# -----------------------
# Check arguments
# -----------------------
if len(sys.argv) != 2:
    print("Usage: python predict.py image.jpg")
    sys.exit(1)

image = cv2.imread(sys.argv[1])
if image is None:
    print("Cannot read image")
    sys.exit(1)
# -----------------------
# Extract Features
# -----------------------

features = extractor.extract(image)
if len(features) != feature_length:
    raise RuntimeError("Feature length mismatch!")

features = features.reshape(1,-1)
# -----------------------
# Predict
# -----------------------

prob = model.predict_proba(features)[0][1]
print(f"{prob:.4f}")