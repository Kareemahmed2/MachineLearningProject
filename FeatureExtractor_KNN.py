"""
Enhanced Feature Extractor for k-NN Classifier
Extracts compact feature vectors optimized for distance-based classification
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy import ndimage
from typing import List

from config import (
    COLOR_HIST_BINS_HSV, LBP_POINTS_MULTIPLIER
)


def extract_color_histogram(img: np.ndarray) -> np.ndarray:
    """
    Extract color histogram in HSV color space.
    HSV is more robust to lighting variations.
    
    Args:
        img: Input BGR image
    
    Returns:
        Normalized color histogram
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3D histogram
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        COLOR_HIST_BINS_HSV,
        [0, 180, 0, 256, 0, 256]
    )
    
    # Normalize
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def extract_lbp_features(img: np.ndarray) -> np.ndarray:
    """
    Extract Local Binary Pattern features.
    Uses uniform LBP for compact representation.
    
    Args:
        img: Input BGR image
    
    Returns:
        LBP histogram
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use radius=2, points=16 for good balance
    radius = 2
    n_points = 16
    
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Histogram
    n_bins = n_points + 2
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )
    
    return hist


def extract_intensity_stats(img: np.ndarray) -> np.ndarray:
    """
    Extract intensity statistics for each color channel.
    
    Args:
        img: Input BGR image
    
    Returns:
        Statistical features (mean, std per channel)
    """
    features = []
    
    for i in range(3):
        channel = img[:, :, i].astype(np.float64)
        features.extend([
            channel.mean() / 255.0,
            channel.std() / 255.0
        ])
    
    return np.array(features)


def extract_edge_distribution(img: np.ndarray) -> np.ndarray:
    """
    Extract edge distribution features.
    Captures edge density in different regions.
    
    Args:
        img: Input BGR image
    
    Returns:
        Edge distribution features
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    h, w = edges.shape
    h2, w2 = h // 2, w // 2
    
    # Quadrant edge densities
    quadrants = [
        edges[:h2, :w2],      # Top-left
        edges[:h2, w2:],      # Top-right
        edges[h2:, :w2],      # Bottom-left
        edges[h2:, w2:]       # Bottom-right
    ]
    
    features = [q.mean() / 255.0 for q in quadrants]
    
    # Overall statistics
    features.extend([
        edges.mean() / 255.0,
        edges.std() / 255.0,
        np.percentile(edges, 75) / 255.0,
        np.percentile(edges, 90) / 255.0
    ])
    
    return np.array(features)


def extract_gabor_features(img: np.ndarray) -> np.ndarray:
    """
    Extract Gabor filter responses for texture analysis.
    Gabor filters capture texture at different scales and orientations.
    
    Args:
        img: Input BGR image
    
    Returns:
        Gabor texture features
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    
    features = []
    
    # Different frequencies and orientations
    frequencies = [0.1, 0.25]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for freq in frequencies:
        for theta in orientations:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                (21, 21),
                sigma=3,
                theta=theta,
                lambd=1.0/freq,
                gamma=0.5,
                psi=0
            )
            
            # Filter image
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            
            # Extract statistics
            features.extend([
                np.abs(filtered).mean(),
                filtered.std()
            ])
    
    # Normalize
    features = np.array(features)
    features = features / (np.linalg.norm(features) + 1e-7)
    
    return features


def extract_shape_features(img: np.ndarray) -> np.ndarray:
    """
    Extract simple shape features.
    
    Args:
        img: Input BGR image
    
    Returns:
        Shape-related features
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(6)
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Features
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    
    # Circularity
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-7)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = float(w) / (h + 1e-7)
    extent = area / (w * h + 1e-7)
    
    # Convex hull
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-7)
    
    return np.array([
        area / (img.shape[0] * img.shape[1]),  # Normalized area
        circularity,
        aspect_ratio,
        extent,
        solidity,
        perimeter / (2 * (img.shape[0] + img.shape[1]))  # Normalized perimeter
    ])


def extract_dominant_colors(img: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Extract dominant colors using k-means clustering.
    
    Args:
        img: Input BGR image
        k: Number of dominant colors
    
    Returns:
        Dominant color features
    """
    # Reshape to pixel list
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    
    # Sort centers by frequency
    unique, counts = np.unique(labels, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    # Get sorted centers and their proportions
    features = []
    for idx in sorted_idx:
        # Normalized color values
        color = centers[idx] / 255.0
        proportion = counts[idx] / len(labels)
        features.extend(color.tolist())
        features.append(proportion)
    
    return np.array(features)


def extract_features_knn(img: np.ndarray) -> np.ndarray:
    """
    Extract compact feature vector for k-NN classification.
    
    Optimized for distance-based classification:
    - Compact representation (fewer dimensions)
    - Normalized features
    - Robust to lighting variations
    
    Args:
        img: Input BGR image (128x128)
    
    Returns:
        Combined feature vector (~120 dimensions)
    """
    features = np.concatenate([
        extract_color_histogram(img),        # 1024 dims -> normalized
        extract_lbp_features(img),           # 18 dims
        extract_intensity_stats(img),        # 6 dims
        extract_edge_distribution(img),      # 8 dims
        extract_gabor_features(img),         # 16 dims
        extract_shape_features(img),         # 6 dims
        extract_dominant_colors(img, k=3)    # 12 dims
    ])
    
    return features


def get_feature_names() -> List[str]:
    """Get names of all features for analysis."""
    return [
        "Color Histogram (HSV)", "LBP", "Intensity Stats",
        "Edge Distribution", "Gabor Texture", "Shape", "Dominant Colors"
    ]


# Test feature extraction
if __name__ == "__main__":
    # Create a test image
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    features = extract_features_knn(test_img)
    print(f"Total feature dimensions: {len(features)}")
    
    # Print individual feature sizes
    print(f"Color Histogram: {len(extract_color_histogram(test_img))}")
    print(f"LBP: {len(extract_lbp_features(test_img))}")
    print(f"Intensity Stats: {len(extract_intensity_stats(test_img))}")
    print(f"Edge Distribution: {len(extract_edge_distribution(test_img))}")
    print(f"Gabor: {len(extract_gabor_features(test_img))}")
    print(f"Shape: {len(extract_shape_features(test_img))}")
    print(f"Dominant Colors: {len(extract_dominant_colors(test_img))}")
