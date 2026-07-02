import cv2
import numpy as np
import random

class Augmentor:

    def __init__(self):
        pass

    def brightness(self, image):
        alpha = random.uniform(0.85, 1.15)
        beta = random.randint(-20, 20)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def rotate(self, image):
        angle = random.uniform(-5, 5)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(
            image,
            M,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

    def gaussian_noise(self, image):
        noise = np.random.normal(0, 6, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def jpeg(self, image):
        quality = random.randint(60, 95)
        _, enc = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return cv2.imdecode(enc, 1)

    def augment(self, image):
        img = image.copy()
        if random.random() < 0.7:
            img = self.brightness(img)
        if random.random() < 0.5:
            img = self.rotate(img)
        if random.random() < 0.5:
            img = self.gaussian_noise(img)
        if random.random() < 0.5:
            img = self.jpeg(img)
        return img