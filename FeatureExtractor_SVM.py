"""
Enhanced Feature Extractor for SVM Classifier - Optimized Version
Balanced feature set for better accuracy
"""

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from typing import List

from config import IMG_SIZE


def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """
    Extract HOG features with optimized parameters for material classification.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optimized HOG parameters for 128x128 images
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),  # Larger cells = fewer features, more robust
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    
    return features


def extract_color_histogram(img: np.ndarray) -> np.ndarray:
    """
    Extract multi-color space histograms.
    """
    features = []
    
    # HSV histogram (robust to lighting)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i, bins in enumerate([16, 16, 8]):  # H, S, V
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 180 if i == 0 else 256])
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        features.extend(hist.flatten())
    
    # LAB histogram (perceptually uniform)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        features.extend(hist.flatten())
    
    return np.array(features)


def extract_lbp_features(img: np.ndarray) -> np.ndarray:
    """
    Extract multi-scale LBP features.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    for radius in [1, 2, 3]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        features.extend(hist)
    
    return np.array(features)


def extract_texture_features(img: np.ndarray) -> np.ndarray:
    """
    Extract texture features using Gabor filters.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    features = []
    
    # Gabor filter bank
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.1, 0.2, 0.4]:
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=3, theta=theta,
                lambd=1.0/freq, gamma=0.5, psi=0
            )
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            features.extend([np.abs(filtered).mean(), filtered.std()])
    
    return np.array(features) / (np.linalg.norm(features) + 1e-7)


def extract_edge_features(img: np.ndarray) -> np.ndarray:
    """
    Extract edge-based features.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Edge histogram in 4 quadrants
    h, w = edges.shape
    h2, w2 = h // 2, w // 2
    quadrants = [
        edges[:h2, :w2], edges[:h2, w2:],
        edges[h2:, :w2], edges[h2:, w2:]
    ]
    
    features = [q.mean() / 255.0 for q in quadrants]
    features.extend([
        edges.mean() / 255.0,
        edges.std() / 255.0,
        magnitude.mean() / 255.0,
        magnitude.std() / 255.0
    ])
    
    return np.array(features)


def extract_shape_features(img: np.ndarray) -> np.ndarray:
    """
    Extract shape descriptors.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Hu moments
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    return hu


def extract_color_statistics(img: np.ndarray) -> np.ndarray:
    """
    Extract color statistics per channel.
    """
    features = []
    
    for color_space, converter in [
        ('BGR', None),
        ('HSV', cv2.COLOR_BGR2HSV),
        ('LAB', cv2.COLOR_BGR2LAB)
    ]:
        converted = cv2.cvtColor(img, converter) if converter else img
        for i in range(3):
            channel = converted[:, :, i].astype(np.float64)
            features.extend([
                channel.mean() / 255.0,
                channel.std() / 255.0,
                np.percentile(channel, 25) / 255.0,
                np.percentile(channel, 75) / 255.0
            ])
    
    return np.array(features)


def extract_features_svm(img: np.ndarray) -> np.ndarray:
    """
    Extract optimized feature vector for SVM classification.
    
    Total dimensions: ~700 (reduced from 8740 for better generalization)
    """
    features = np.concatenate([
        extract_hog_features(img),           # ~324 dims
        extract_color_histogram(img),        # ~88 dims
        extract_lbp_features(img),           # ~37 dims
        extract_texture_features(img),       # ~24 dims (normalized)
        extract_edge_features(img),          # ~8 dims
        extract_shape_features(img),         # 7 dims
        extract_color_statistics(img)        # ~36 dims
    ])
    
    return features


# Test
if __name__ == "__main__":
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    features = extract_features_svm(test_img)
    print(f"Total feature dimensions: {len(features)}")
