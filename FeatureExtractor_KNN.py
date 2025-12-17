import numpy as np
import cv2
from skimage.feature import local_binary_pattern


def extract_color_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [6, 6],        # compact for distance metrics
        [0,180,0,256]
    )
    return cv2.normalize(hist, hist).flatten()


def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 16, 2, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=18,
        range=(0, 18),
        density=True
    )
    return hist


def extract_intensity_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.array([
        gray.mean(),
        gray.std()
    ])


def extract_edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return np.array([edges.mean()])


def extract_features_knn(img):
    return np.concatenate([
        extract_color_hist(img),
        extract_lbp(img),
        extract_intensity_stats(img),
        extract_edge_density(img)
    ])
