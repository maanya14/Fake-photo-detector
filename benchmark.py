import time
import cv2
import glob
import joblib
from feature_extractor import FeatureExtractor

metadata=joblib.load("models/model.pkl")
model=metadata["model"]
extractor=FeatureExtractor()
images = glob.glob("dataset/real/*")

if len(images) == 0:
    raise FileNotFoundError("No images found in dataset/real")

image = cv2.imread(images[0])

# warmup

for _ in range(10):
    f=extractor.extract(image)
    model.predict_proba(f.reshape(1,-1))
runs=100

start=time.perf_counter()

for _ in range(runs):
    f=extractor.extract(image)
    model.predict_proba(f.reshape(1,-1))

end=time.perf_counter()
latency=((end-start)/runs)*1000
print()
print(f"Average latency : {latency:.2f} ms/image")