"""
Enhanced Feature Extractor for SVM Classifier
Extracts comprehensive feature vectors optimized for SVM classification
"""

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk
from typing import List, Tuple

from config import (
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
    LBP_RADIUS, LBP_POINTS_MULTIPLIER, COLOR_HIST_BINS_LAB
)


def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    HOG captures edge directions and is excellent for shape recognition.
    
    Args:
        img: Input BGR image
    
    Returns:
        HOG feature vector
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    
    return features


def extract_color_histogram(img: np.ndarray) -> np.ndarray:
    """
    Extract color histogram in LAB color space.
    LAB is perceptually uniform and better for material differentiation.
    
    Args:
        img: Input BGR image
    
    Returns:
        Normalized color histogram
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Calculate 3D histogram
    hist = cv2.calcHist(
        [lab],
        [0, 1, 2],
        None,
        COLOR_HIST_BINS_LAB,
        [0, 256, 0, 256, 0, 256]
    )
    
    # Normalize and flatten
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def extract_lbp_features(img: np.ndarray) -> np.ndarray:
    """
    Extract multi-scale Local Binary Pattern (LBP) features.
    LBP captures texture information at different scales.
    
    Args:
        img: Input BGR image
    
    Returns:
        LBP histogram features
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    for radius in LBP_RADIUS:
        n_points = LBP_POINTS_MULTIPLIER * radius
        
        # Compute LBP
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Compute histogram
        n_bins = n_points + 2  # +2 for uniform LBP
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )
        features.extend(hist)
    
    return np.array(features)


def extract_glcm_features(img: np.ndarray) -> np.ndarray:
    """
    Extract Gray-Level Co-occurrence Matrix (GLCM) features.
    GLCM captures texture patterns based on pixel pair statistics.
    
    Args:
        img: Input BGR image
    
    Returns:
        GLCM texture features (contrast, homogeneity, energy, correlation)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Reduce gray levels for faster computation
    gray_reduced = (gray // 16).astype(np.uint8)
    
    # Compute GLCM for multiple angles
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(
        gray_reduced,
        distances=distances,
        angles=angles,
        levels=16,
        symmetric=True,
        normed=True
    )
    
    # Extract properties
    properties = ['contrast', 'homogeneity', 'energy', 'correlation']
    features = []
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        features.extend(values.flatten())
    
    return np.array(features)


def extract_entropy_features(img: np.ndarray) -> np.ndarray:
    """
    Extract local entropy features.
    Entropy measures texture complexity/randomness.
    
    Args:
        img: Input BGR image
    
    Returns:
        Entropy statistics (mean, std, min, max)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Local entropy with disk structuring element
    ent = entropy(gray, disk(5))
    
    return np.array([
        ent.mean(),
        ent.std(),
        ent.min(),
        ent.max()
    ])


def extract_edge_features(img: np.ndarray) -> np.ndarray:
    """
    Extract edge-based features using Canny edge detection.
    
    Args:
        img: Input BGR image
    
    Returns:
        Edge statistics and directional features
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny edges at different thresholds
    edges_low = cv2.Canny(gray, 30, 100)
    edges_high = cv2.Canny(gray, 100, 200)
    
    # Sobel gradients for direction
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and direction
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    return np.array([
        edges_low.mean() / 255.0,
        edges_high.mean() / 255.0,
        edges_low.std() / 255.0,
        magnitude.mean() / 255.0
    ])


def extract_hu_moments(img: np.ndarray) -> np.ndarray:
    """
    Extract Hu moments for shape invariance.
    Hu moments are invariant to translation, scale, and rotation.
    
    Args:
        img: Input BGR image
    
    Returns:
        Log-transformed Hu moments
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate moments
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform for better scale
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments


def extract_color_moments(img: np.ndarray) -> np.ndarray:
    """
    Extract color moments (mean, std, skewness) for each channel.
    
    Args:
        img: Input BGR image
    
    Returns:
        Color moment features
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    features = []
    
    for color_img in [img, hsv, lab]:
        for i in range(3):
            channel = color_img[:, :, i].astype(np.float64)
            
            # Mean
            mean = np.mean(channel)
            # Standard deviation
            std = np.std(channel)
            # Skewness
            skewness = np.mean(((channel - mean) / (std + 1e-7)) ** 3)
            
            features.extend([mean / 255.0, std / 255.0, skewness])
    
    return np.array(features)


def extract_features_svm(img: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive feature vector for SVM classification.
    
    Combines multiple feature types:
    - HOG: Shape and edge orientation
    - Color Histogram: Color distribution
    - LBP: Texture patterns
    - GLCM: Texture statistics
    - Entropy: Complexity measure
    - Edge: Edge density and direction
    - Hu Moments: Shape invariants
    - Color Moments: Color statistics
    
    Args:
        img: Input BGR image (128x128)
    
    Returns:
        Combined feature vector (~2300 dimensions)
    """
    features = np.concatenate([
        extract_hog_features(img),           # ~1764 dims
        extract_color_histogram(img),        # 512 dims
        extract_lbp_features(img),           # ~37 dims
        extract_glcm_features(img),          # 32 dims
        extract_entropy_features(img),       # 4 dims
        extract_edge_features(img),          # 4 dims
        extract_hu_moments(img),             # 7 dims
        extract_color_moments(img)           # 27 dims
    ])
    
    return features


def get_feature_names() -> List[str]:
    """Get names of all features for analysis."""
    return [
        "HOG", "Color Histogram (LAB)", "LBP (multi-scale)",
        "GLCM", "Entropy", "Edge", "Hu Moments", "Color Moments"
    ]


# Test feature extraction
if __name__ == "__main__":
    # Create a test image
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    features = extract_features_svm(test_img)
    print(f"Total feature dimensions: {len(features)}")
    
    # Print individual feature sizes
    print(f"HOG: {len(extract_hog_features(test_img))}")
    print(f"Color Histogram: {len(extract_color_histogram(test_img))}")
    print(f"LBP: {len(extract_lbp_features(test_img))}")
    print(f"GLCM: {len(extract_glcm_features(test_img))}")
    print(f"Entropy: {len(extract_entropy_features(test_img))}")
    print(f"Edge: {len(extract_edge_features(test_img))}")
    print(f"Hu Moments: {len(extract_hu_moments(test_img))}")
    print(f"Color Moments: {len(extract_color_moments(test_img))}")
