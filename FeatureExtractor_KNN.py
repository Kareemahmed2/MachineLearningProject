import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew


def extract_color_hist(img):
    """Enhanced color histograms - more compact for k-NN"""
    features = []

    # HSV histogram (good for material classification)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [6, 6, 6],
                            [0, 180, 0, 256, 0, 256])
    features.extend(cv2.normalize(hist_hsv, hist_hsv).flatten())

    # LAB histogram
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist_lab = cv2.calcHist([lab], [1, 2], None, [8, 8],
                            [0, 256, 0, 256])
    features.extend(cv2.normalize(hist_lab, hist_lab).flatten())

    return np.array(features)


def extract_color_moments(img):
    """Statistical color moments"""
    features = []

    # HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in cv2.split(hsv):
        features.extend([
            channel.mean(),
            channel.std(),
            skew(channel.flatten())
        ])

    # LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for channel in cv2.split(lab):
        features.extend([
            channel.mean(),
            channel.std()
        ])

    return np.array(features)


def extract_lbp(img):
    """Multi-scale LBP for texture"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []

    for radius in [1, 2]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_points + 2,
            range=(0, n_points + 2),
            density=True
        )
        features.extend(hist)

    return np.array(features)


def extract_glcm_features(img):
    """GLCM texture features - compact version"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Quantize to reduce computation
    gray_quantized = (gray / 32).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        gray_quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True
    )

    features = []
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        features.extend(graycoprops(glcm, prop).flatten())

    return np.array(features)


def extract_intensity_stats(img):
    """Enhanced intensity statistics"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.array([
        gray.mean(),
        gray.std(),
        np.median(gray),
        np.percentile(gray, 25),
        np.percentile(gray, 75),
        skew(gray.flatten())
    ])


def extract_edge_density(img):
    """Multi-threshold edge features"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []

    # Multiple Canny thresholds
    for low, high in [(50, 150), (80, 200)]:
        edges = cv2.Canny(gray, low, high)
        features.extend([
            edges.mean(),
            np.count_nonzero(edges) / edges.size
        ])

    return np.array(features)


def extract_gradient_features(img):
    """Gradient magnitude statistics"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    return np.array([
        magnitude.mean(),
        magnitude.std(),
        magnitude.max()
    ])


def extract_shape_features(img):
    """Basic shape descriptors"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # Hu moments (compact shape descriptors)
        moments = cv2.moments(largest)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        return np.concatenate([
            [area / (gray.shape[0] * gray.shape[1]), circularity],
            hu_log[:5]  # First 5 Hu moments
        ])
    else:
        return np.zeros(7)


def extract_features_knn(img):

    return np.concatenate([
        extract_color_hist(img),
        extract_color_moments(img),
        extract_lbp(img),
        extract_glcm_features(img),
        extract_intensity_stats(img),
        extract_edge_density(img),
        extract_gradient_features(img),
        extract_shape_features(img)
    ])