import numpy as np
import cv2
from skimage.feature import hog

def extract_hog_single(img, pixels_per_cell):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True
    )

def extract_color_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    )
    return cv2.normalize(hist, hist).flatten()

def extract_features_svm(img):
    hog_8  = extract_hog_single(img, (8, 8))    # fine texture
    hog_16 = extract_hog_single(img, (16, 16))  # structure

    color = extract_color_hist(img)

    return np.concatenate([
        hog_8,
        hog_16,
        color
    ])
