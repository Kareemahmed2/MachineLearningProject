"""
Enhanced Feature Extractor for k-NN Classifier - Optimized Version
Compact features optimized for distance-based classification
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from typing import List

from config import IMG_SIZE


def extract_color_histogram(img: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive color histograms.
    """
    features = []
    
    # HSV histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i, bins in enumerate([18, 16, 8]):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 180 if i == 0 else 256])
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        features.extend(hist.flatten())
    
    # LAB histogram
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [12], [0, 256])
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        features.extend(hist.flatten())
    
    return np.array(features)


def extract_lbp_features(img: np.ndarray) -> np.ndarray:
    """
    Extract LBP features at multiple scales.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    for radius in [1, 2]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        features.extend(hist)
    
    return np.array(features)


def extract_texture_energy(img: np.ndarray) -> np.ndarray:
    """
    Extract texture energy using Gabor filters.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    features = []
    
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.15, 0.3]:
            kernel = cv2.getGaborKernel(
                (15, 15), sigma=2.5, theta=theta,
                lambd=1.0/freq, gamma=0.5, psi=0
            )
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            features.extend([np.abs(filtered).mean(), filtered.std()])
    
    # Normalize
    features = np.array(features)
    return features / (np.linalg.norm(features) + 1e-7)


def extract_edge_distribution(img: np.ndarray) -> np.ndarray:
    """
    Extract edge distribution features.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    h, w = edges.shape
    h2, w2 = h // 2, w // 2
    
    # Quadrant densities
    features = [
        edges[:h2, :w2].mean() / 255.0,
        edges[:h2, w2:].mean() / 255.0,
        edges[h2:, :w2].mean() / 255.0,
        edges[h2:, w2:].mean() / 255.0,
        edges.mean() / 255.0,
        edges.std() / 255.0
    ]
    
    return np.array(features)


def extract_color_moments(img: np.ndarray) -> np.ndarray:
    """
    Extract color moments (mean, std, skewness).
    """
    features = []
    
    for converter in [None, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB]:
        converted = cv2.cvtColor(img, converter) if converter else img
        for i in range(3):
            channel = converted[:, :, i].astype(np.float64)
            mean = channel.mean()
            std = channel.std()
            # Skewness
            skew = np.mean(((channel - mean) / (std + 1e-7)) ** 3)
            features.extend([mean / 255.0, std / 255.0, skew / 10.0])
    
    return np.array(features)


def extract_dominant_colors(img: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Extract dominant colors using k-means.
    """
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    
    # Sort by frequency
    unique, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    features = []
    for idx in sorted_idx:
        color = centers[idx] / 255.0
        proportion = counts[idx] / len(labels)
        features.extend(color.tolist())
        features.append(proportion)
    
    return np.array(features)


def extract_spatial_features(img: np.ndarray) -> np.ndarray:
    """
    Extract spatial pyramid features.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    # Global stats
    features.extend([gray.mean() / 255.0, gray.std() / 255.0])
    
    # 2x2 grid stats
    h, w = gray.shape
    h2, w2 = h // 2, w // 2
    for region in [gray[:h2, :w2], gray[:h2, w2:], gray[h2:, :w2], gray[h2:, w2:]]:
        features.extend([region.mean() / 255.0, region.std() / 255.0])
    
    return np.array(features)


def extract_features_knn(img: np.ndarray) -> np.ndarray:
    """
    Extract optimized feature vector for k-NN classification.
    
    Compact representation for reliable distance computation.
    Total dimensions: ~150
    """
    features = np.concatenate([
        extract_color_histogram(img),        # ~78 dims
        extract_lbp_features(img),           # ~28 dims
        extract_texture_energy(img),         # ~16 dims (normalized)
        extract_edge_distribution(img),      # 6 dims
        extract_color_moments(img),          # ~27 dims
        extract_dominant_colors(img, k=3),   # 12 dims
        extract_spatial_features(img)        # 10 dims
    ])
    
    return features


# Test
if __name__ == "__main__":
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    features = extract_features_knn(test_img)
    print(f"Total feature dimensions: {len(features)}")
