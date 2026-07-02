import os
import cv2
import joblib
import numpy as n
from tqdm import tqdm
from utils import image_paths
from augment import Augmentor
from feature_extractor import FeatureExtractor

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# ---------------------------------------

DATASET = "dataset"

REAL = os.path.join(DATASET, "real")
SCREEN = os.path.join(DATASET, "screen")
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)
extractor = FeatureExtractor()
augmentor = Augmentor()

# ---------------------------------------

X = []
y = []

print("Loading REAL images...")

for file in tqdm(image_paths(REAL)):
    img = cv2.imread(file)
    X.append(extractor.extract(img))
    y.append(0)

print("Loading SCREEN images...")

for file in tqdm(image_paths(SCREEN)):
    img = cv2.imread(file)
    X.append(extractor.extract(img))
    y.append(1)

X = n.array(X)
y = n.array(y)
print()
print("Images :", len(X))
print("Features :", X.shape[1])
print()

# ---------------------------------------

models = {

    "Random Forest":
    RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        bootstrap=True,
        max_features="sqrt",
        n_jobs=-1
    ),

    "SVM":
    Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(
            probability=True,
            kernel="rbf",
            C=5,
            gamma=0.02
        ))
    ]),

    "Logistic":
    Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000
        ))
    ]),

    "Extra Trees":
    ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        max_features="sqrt",
        n_jobs=-1
    ),

    "Hist Gradient Boosting":
    HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=4,
        l2_regularization=1.0,
        random_state=42
    ),

    "Gradient Boosting":
    GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
}

# ---------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

best_score = -1
best_name = None
best_model = None
print("Cross Validation\n")
for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    print(name)

    print(f"Scores : {scores}")

    print(f"Mean : {scores.mean():.4f}")

    print(f"Std  : {scores.std():.4f}")

    print()
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_name = name
        best_model = model
print("-----------------------------------")
print("Best :", best_name)
print("CV Accuracy :", best_score)
print("-----------------------------------")
print("\nCreating augmented training set...")

X_final = []
y_final = []

# REAL
for file in image_paths(REAL):
    img = cv2.imread(file)
    X_final.append(extractor.extract(img))
    y_final.append(0)

    for _ in range(3):        # 3 augmented versions
        aug = augmentor.augment(img)
        X_final.append(extractor.extract(aug))
        y_final.append(0)

# SCREEN
for file in image_paths(SCREEN):
    img = cv2.imread(file)
    X_final.append(extractor.extract(img))
    y_final.append(1)
    for _ in range(3):
        aug = augmentor.augment(img)
        X_final.append(extractor.extract(aug))
        y_final.append(1)

X_final = n.array(X_final)
y_final = n.array(y_final)

print(f"Final training samples: {len(X_final)}")

best_model.fit(X_final, y_final)

metadata = {
    "model": best_model,
    "feature_names": extractor.feature_names,
    "feature_length": len(extractor.extract(cv2.imread(image_paths(REAL)[0]))),
    "model_name": best_name,
    "cv_accuracy": float(best_score)
}

joblib.dump(
    metadata,
    os.path.join(
        MODEL_DIR,
        "model.pkl"
    )
)

print()
print("Model Saved")

# ---------------------------------------

if best_name == "Random Forest":
    importances = best_model.feature_importances_
    names = extractor.feature_names
    idx = n.argsort(importances)[::-1]
    print()
    print("Top Important Features")
    print("----------------------------")
    for i in idx[:15]:
        print(
            f"{names[i]:25s}"
            f"{importances[i]:.4f}"
        )