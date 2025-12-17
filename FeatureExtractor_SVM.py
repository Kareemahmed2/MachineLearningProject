import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk


def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True
    )


def extract_color_hist(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist(
        [lab],
        [0, 1, 2],
        None,
        [8, 8, 4],      # 256 bins (SVM-friendly)
        [0,180,0,256,0,256]
    )
    return cv2.normalize(hist, hist).flatten()


def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []

    for radius in [1, 2]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_points + 2,
            range=(0, n_points + 2),
            density=True
        )
        feats.extend(hist)

    return np.array(feats)


def extract_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ent = entropy(gray, disk(3))
    return np.array([ent.mean(), ent.std()])


def extract_edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.array([edges.mean()])


def extract_features_svm(img):
    return np.concatenate([
        extract_hog(img),
        extract_color_hist(img),
        extract_lbp(img),
        extract_entropy(img),
        extract_edge_density(img)
    ])
