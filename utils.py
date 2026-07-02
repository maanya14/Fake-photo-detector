import cv2
import numpy as np
import os

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def load_image(path):

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image : {path}")
    return img


def image_paths(folder):
    files = []
    for file in os.listdir(folder):
        if file.lower().endswith(VALID_EXTENSIONS):
            files.append(os.path.join(folder,file))

    files.sort()
    return files


def normalize(v):
    v=np.array(v,dtype=np.float32)
    return (v-v.mean())/(v.std()+1e-8)