import cv2
import numpy as np
from scipy.stats import entropy
from skimage.feature import local_binary_pattern

LBP_RADIUS=2
LBP_POINTS=16


class FeatureExtractor:
    def __init__(self,size=(256,256)):
        self.size=size
        self._radius_cache={}
        self._theta_cache={}

    ##################################################

    @property
    def feature_names(self):
        names=[]
        names.extend([
            "laplacian",
            "tenengrad",
            "edge_density",
            "edge_mean",
            "edge_std",
            "fft_high_ratio",
            "fft_entropy",
            "fft_radial_std",
            "moire_peak_count",
            "moire_peak_strength",
            "fft_angular_anisotropy",
            "fft_angular_cv",
            "entropy",
            "gray_mean",
            "gray_std",
            "bright_ratio",
            "bright_mean",
            "bright_std",
            "patch_lap_mean",
            "patch_lap_std",
            "patch_edge_mean",
            "patch_edge_std",
            "patch_fft_mean",
            "patch_fft_std"
        ])
        rgb=["B","G","R"]
        for c in rgb:
            names.append(f"{c}_mean")
            names.append(f"{c}_std")
        hsv=["H","S","V"]
        for c in hsv:
            names.append(f"{c}_mean")
            names.append(f"{c}_std")
        for i in range(LBP_POINTS+2):
            names.append(f"lbp_{i}")
        names.extend([
            "lbp_mean",
            "lbp_std"
        ])
        return names

    ##################################################

    def preprocess(self,image):
        image=cv2.resize(image,self.size)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image,gray

    ##################################################

    def laplacian(self,gray):
        return cv2.Laplacian(gray,cv2.CV_64F).var()

    ##################################################

    def tenengrad(self,gray):
        gx=cv2.Sobel(gray,cv2.CV_64F,1,0)
        gy=cv2.Sobel(gray,cv2.CV_64F,0,1)
        return np.mean(np.sqrt(gx**2+gy**2))

    ##################################################

    def edge_features(self,gray):
        edges=cv2.Canny(gray,100,200)
        density=np.mean(edges>0)
        gx=cv2.Sobel(gray,cv2.CV_64F,1,0)
        gy=cv2.Sobel(gray,cv2.CV_64F,0,1)
        mag=np.sqrt(gx**2+gy**2)

        return [
            density,
            mag.mean(),
            mag.std()
        ]

    ##################################################

    def patch_statistics(self, gray):
        h, w = gray.shape
        ph = h // 3
        pw = w // 3
        lap = []
        edge = []
        fft = []
        for i in range(3):
            for j in range(3):
                patch = gray[
                    i*ph:(i+1)*ph,
                    j*pw:(j+1)*pw
                ]
                lap.append(self.laplacian(patch))
                edge.append(self.edge_features(patch)[0])
                fft.append(self.fft_features(patch)[0])
        return [

            np.mean(lap),
            np.std(lap),

            np.mean(edge),
            np.std(edge),

            np.mean(fft),
            np.std(fft)
        ]
    
    ##################################################

    def _radius_map(self, shape):
        key = shape
        if key not in self._radius_cache:
            h, w = shape
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.int32)
            self._radius_cache[key] = radius
        return self._radius_cache[key]

    def _theta_map(self, shape):
        key = shape
        if key not in self._theta_cache:
            h, w = shape
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            theta = np.arctan2(Y - cy, X - cx)
            self._theta_cache[key] = theta
        return self._theta_cache[key]

    def fft_features(self, gray):

        f = np.fft.fft2(gray)
        f = np.fft.fftshift(f)
        mag = np.log1p(np.abs(f))
        h, w = gray.shape
        radius = self._radius_map((h, w))

        max_radius = int(radius.max())
        # vectorized radial average (replaces the old per-radius python loop)
        sums = np.bincount(radius.ravel(), weights=mag.ravel(), minlength=max_radius + 1)
        counts = np.bincount(radius.ravel(), minlength=max_radius + 1)
        radial = sums[:max_radius] / np.maximum(counts[:max_radius], 1)

        high = radial[len(radial)//2:]
        high_ratio = np.sum(high) / (np.sum(radial) + 1e-8)
        spectral_entropy = entropy(
            radial / (radial.sum() + 1e-8)
        )
        radial_std = np.std(radial)

        # -------- Moiré Peak Detector --------

        mean = high.mean()
        std = high.std()
        threshold = mean + 2 * std
        peaks = high[high > threshold]
        peak_count = len(peaks)
        if peak_count > 0:
            peak_strength = peaks.mean()
        else:
            peak_strength = 0
        return [
            high_ratio,
            spectral_entropy,
            radial_std,
            peak_count,
            peak_strength
        ]

    ##################################################
    # Screens expose a periodic RGB subpixel / pixel grid. That grid shows up
    # as energy concentrated at specific ANGLES in the FFT (not just "more
    # high frequency energy" overall, which the radial profile above already
    # captures). Real-world textures are much more angularly uniform at the
    # same frequency band. This measures that anisotropy.
    def angular_features(self, gray):
        f = np.fft.fft2(gray)
        f = np.fft.fftshift(f)
        mag = np.log1p(np.abs(f))
        h, w = gray.shape
        radius = self._radius_map((h, w))
        theta = self._theta_map((h, w))

        max_radius = radius.max()
        r_lo = 0.12 * max_radius
        r_hi = 0.48 * max_radius
        band = (radius >= r_lo) & (radius <= r_hi)

        n_bins = 24
        theta_idx = ((theta[band] + np.pi) / (2 * np.pi) * n_bins).astype(np.int32)
        theta_idx = np.clip(theta_idx, 0, n_bins - 1)

        sums = np.bincount(theta_idx, weights=mag[band], minlength=n_bins)
        counts = np.bincount(theta_idx, minlength=n_bins)
        profile = sums / np.maximum(counts, 1)

        mean = profile.mean()
        std = profile.std()
        peak = profile.max()
        anisotropy = (peak - mean) / (mean + 1e-8)
        cv = std / (mean + 1e-8)
        return [anisotropy, cv]

    ##################################################

    def entropy_features(self,gray):
        hist=cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0,256]
        )

        hist/=hist.sum()
        return [
            entropy(hist.flatten()+1e-8),
            gray.mean(),
            gray.std()
        ]
    ##################################################

    def bright_features(self,gray):

        bright=gray>240
        ratio=np.mean(bright)
        if bright.any():
            mean=gray[bright].mean()
            std=gray[bright].std()
        else:
            mean=0
            std=0
        return [
            ratio,
            mean,
            std
        ]
    ##################################################

    def rgb_features(self,image):
        features=[]
        for i in range(3):
            c=image[:,:,i]
            features.extend([
                c.mean(),
                c.std()
            ])
        return features

    ##################################################

    def hsv_features(self,image):
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        H,S,V=cv2.split(hsv)
        return [
            H.mean(),H.std(),
            S.mean(),S.std(),
            V.mean(),V.std()
        ]

    ##################################################

    def lbp_features(self,gray):
        lbp=local_binary_pattern(
            gray,
            LBP_POINTS,
            LBP_RADIUS,
            method="uniform"
        )

        hist,_=np.histogram(
            lbp.ravel(),
            bins=np.arange(0,LBP_POINTS+3),
            density=True
        )
        hist=list(hist)
        hist.append(lbp.mean())
        hist.append(lbp.std())
        return hist
    ##################################################
    
    def extract(self,image):
        image,gray=self.preprocess(image)
        features=[]
        features.append(self.laplacian(gray))
        features.append(self.tenengrad(gray))
        features.extend(self.edge_features(gray))
        features.extend(self.fft_features(gray))
        features.extend(self.angular_features(gray))
        features.extend(self.entropy_features(gray))
        features.extend(self.bright_features(gray))
        features.extend(self.rgb_features(image))
        features.extend(self.hsv_features(image))
        features.extend(self.lbp_features(gray))
        features.extend(
            self.patch_statistics(gray)
        )
        return np.array(features,dtype=np.float32)