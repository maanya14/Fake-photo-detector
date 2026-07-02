# Fake Photo Detector

Given a photo, decide whether it's a **real photo** of a real thing, or a **recapture** — a photo taken of a phone/laptop screen or a printout. Outputs a single score from 0 (real) to 1 (screen).

## Approach

No deep learning — this uses classic computer-vision / signal-processing features fed into a small classical ML model, so it stays fast and small enough to eventually run on a phone.

**Features (~56 per image):**
- **Frequency domain:** radial FFT profile (high-frequency energy ratio, spectral entropy, moiré peak detection) plus an **angular FFT anisotropy** feature that specifically targets the periodic RGB subpixel/pixel grid a screen produces — real-world textures are far more angularly uniform at the same frequency band.
- **Sharpness / texture:** Laplacian variance, Tenengrad (Sobel gradient magnitude), Canny edge density, Local Binary Patterns (LBP).
- **Patch statistics:** the same sharpness/edge/FFT signals recomputed over a 3x3 grid of patches, since screen artifacts (glare, moiré) are often localized rather than global.
- **Color:** RGB and HSV channel statistics, overexposed/bright-pixel ratio (screen glare).

**Model:** RBF-kernel SVM (`C=5, gamma=0.02`), selected via repeated stratified 5-fold cross-validation against Random Forest, Extra Trees, Logistic Regression, and Gradient Boosting variants. The final model is fit on the 104 base photos plus 3 augmented copies of each (brightness jitter, small rotation, Gaussian noise, JPEG re-compression) — 416 samples total.

## Results 

| Metric | Value |
|---|---|
| Accuracy (0.5 threshold) | 0.856 |
| Accuracy (tuned threshold = 0.38) | **0.875** |
| ROC AUC | 0.934 |
| Latency | ~43 ms/image (laptop CPU) |
| Cost | $0 on-device; ~$0.0002-0.0003 per 1,000 images on a cheap cloud CPU instance |

All numbers come from `cross_val_predict` averaged over 5 random splits, so the model is always scored on folds it never trained on. See `evaluate.py`.

**Below the 95% target** — the limiting factor is dataset size (52 real / 52 screen photos), not the approach. See "What I'd improve" below.

## What I'd improve with more time

1. **More data.** 150-200+ photos per class across more screens (OLED/LCD, phone/laptop/TV), lighting conditions, and angles — CV variance is still ~6% at the current sample size, so this is the highest-leverage next step.
2. **More features:** chromatic fringing at edges, specular-glare blob shape/compactness.
3. **A CNN**, once there's enough data to support one without overfitting — likely a small quantized MobileNet-scale model to keep it phone-friendly.
4. **Adversarial robustness:** keep a feedback loop on flagged/borderline cases and periodically retrain as new screen types appear, since moiré-based cues are the easiest for a cheater to defeat with a matte screen protector.

## Project structure

```
predict.py             # one-line predictor: python predict.py image.jpg -> 0.93
train.py               # trains + selects the model, saves models/model.pkl
evaluate.py            # honest cross-validated accuracy/precision/recall/F1/ROC-AUC + confusion matrix
benchmark.py            # measures per-image latency
feature_extractor.py    # all feature engineering
augment.py               # training-time augmentation
utils.py                  # small IO helpers
dataset/real/              # real photos
dataset/screen/             # screen/printout recaptures
models/model.pkl              # trained model + metadata
```

## Usage

```bash
pip install -r requirements.txt

python train.py       # retrain from dataset/ -> models/model.pkl
python evaluate.py    # honest held-out metrics + confusion_matrix.png
python benchmark.py   # per-image latency
python predict.py path/to/image.jpg   # -> 0.0-1.0 score
```

## Live demo (optional, camera-based)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens a browser tab that streams your webcam through the model in real time and overlays the REAL / SCREEN label and score on the video. All processing happens locally — nothing is uploaded. Threshold and frame-skip rate are adjustable from the sidebar.


