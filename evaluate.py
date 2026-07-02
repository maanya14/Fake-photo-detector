import os
import cv2
import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import *

from feature_extractor import FeatureExtractor
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from utils import image_paths

DATASET="dataset"
REAL=os.path.join(DATASET,"real")
SCREEN=os.path.join(DATASET,"screen")
metadata=joblib.load("models/model.pkl")
model=metadata["model"]
extractor=FeatureExtractor()

X=[]
y=[]


for file in image_paths(REAL):
    img=cv2.imread(file)
    X.append(extractor.extract(img))
    y.append(0)

for file in image_paths(SCREEN):
    img=cv2.imread(file)
    X.append(extractor.extract(img))
    y.append(1)

X=np.array(X)
y=np.array(y)

SEEDS = [42, 1, 7, 13, 99]
prob_sum = np.zeros(len(y))
for seed in SEEDS:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    prob_sum += p
prob = prob_sum / len(SEEDS)
pred = (prob >= 0.5).astype(int)

print("Accuracy :",accuracy_score(y,pred))
print("Precision:",precision_score(y,pred))
print("Recall :",recall_score(y,pred))
print("F1 :",f1_score(y,pred))
print("ROC AUC :",roc_auc_score(y,prob))
print()

best_t, best_acc = 0.5, 0
for t in np.arange(0.05, 0.96, 0.01):
    acc = accuracy_score(y, (prob >= t).astype(int))
    if acc > best_acc:
        best_acc, best_t = acc, t
print(f"Optimal threshold (from CV probabilities): {best_t:.2f}")
print(f"Accuracy at that threshold             : {best_acc:.4f}")
print()

cm=confusion_matrix(y,pred)
disp=ConfusionMatrixDisplay(cm)
disp.plot()
plt.tight_layout()
plt.savefig("confusion_matrix.png",dpi=300)