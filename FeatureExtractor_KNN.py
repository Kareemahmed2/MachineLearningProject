"""
Powerful Feature Extractor for k-NN - Version 3
Compact but discriminative features
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from typing import List


def extract_color_histogram(img: np.ndarray) -> np.ndarray:
    """Multi-space color histograms."""
    features = []
    
    # HSV histogram (H is circular, important for color)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i, bins in enumerate([24, 12, 8]):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 180 if i==0 else 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    # LAB histogram
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [12], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    return np.array(features)


def extract_texture_features(img: np.ndarray) -> np.ndarray:
    """Texture analysis."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    # LBP at multiple scales
    for radius in [1, 2]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2), density=True)
        features.extend(hist)
    
    # Gabor features
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.15, 0.3]:
            kernel = cv2.getGaborKernel((15, 15), 3.0, theta, 1/freq, 0.5, 0)
            filtered = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kernel)
            features.extend([np.abs(filtered).mean()/100, filtered.std()/100])
    
    return np.array(features)


def extract_edge_features(img: np.ndarray) -> np.ndarray:
    """Edge distribution."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    h, w = edges.shape
    features = []
    
    # Quadrant densities
    h2, w2 = h//2, w//2
    for region in [edges[:h2,:w2], edges[:h2,w2:], edges[h2:,:w2], edges[h2:,w2:]]:
        features.append(region.mean()/255)
    
    # Global edge stats
    features.extend([edges.mean()/255, edges.std()/128])
    
    # Gradient direction histogram
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    direction = np.arctan2(sobely, sobelx)
    dir_hist, _ = np.histogram(direction, bins=8, range=(-np.pi, np.pi), density=True)
    features.extend(dir_hist)
    
    return np.array(features)


def extract_color_moments(img: np.ndarray) -> np.ndarray:
    """Statistical color features."""
    features = []
    
    for space, converter in [('BGR', None), ('HSV', cv2.COLOR_BGR2HSV), ('LAB', cv2.COLOR_BGR2LAB)]:
        converted = cv2.cvtColor(img, converter) if converter else img
        for i in range(3):
            ch = converted[:,:,i].astype(np.float64)
            mean = ch.mean()
            std = ch.std() + 1e-7
            skew = np.mean(((ch - mean)/std)**3)
            features.extend([mean/255, std/128, skew/5])
    
    return np.array(features)


def extract_reflectance_features(img: np.ndarray) -> np.ndarray:
    """Material reflectance properties."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    features = []
    
    # Brightness analysis
    v = hsv[:,:,2]
    s = hsv[:,:,1]
    
    features.append((v > 200).mean())  # Specular highlights
    features.append((s < 30).mean())   # Metallic/grey areas
    features.append(gray.std()/128)     # Contrast
    
    # Laplacian for edge sharpness
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.abs(lap).mean()/50)
    
    return np.array(features)


def extract_dominant_colors(img: np.ndarray, k: int = 4) -> np.ndarray:
    """K-means dominant colors."""
    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    
    unique, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    features = []
    for idx in sorted_idx:
        features.extend((centers[idx]/255).tolist())
        features.append(counts[idx]/len(labels))
    
    return np.array(features)


def extract_features_knn(img: np.ndarray) -> np.ndarray:
    """
    Compact feature vector for k-NN.
    """
    features = np.concatenate([
        extract_color_histogram(img),
        extract_texture_features(img),
        extract_edge_features(img),
        extract_color_moments(img),
        extract_reflectance_features(img),
        extract_dominant_colors(img, k=4)
    ])
    
    return features


if __name__ == "__main__":
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    features = extract_features_knn(test_img)
    print(f"Total feature dimensions: {len(features)}")
