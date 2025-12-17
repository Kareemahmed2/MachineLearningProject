"""
Powerful Feature Extractor for SVM - Version 3
Optimized for material discrimination
"""

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from scipy import ndimage
from typing import List


def extract_hog_multiscale(img: np.ndarray) -> np.ndarray:
    """Multi-scale HOG features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    # Multiple cell sizes for multi-scale
    for ppc in [(8, 8), (16, 16)]:
        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=ppc,
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        features.extend(hog_feat)
    
    return np.array(features)


def extract_color_features(img: np.ndarray) -> np.ndarray:
    """Comprehensive color features."""
    features = []
    
    # HSV - good for color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i, (bins, range_max) in enumerate([(32, 180), (16, 256), (16, 256)]):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, range_max])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    # LAB - perceptually uniform
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    
    # Color moments (mean, std, skewness) per channel
    for color in [img, hsv, lab]:
        for i in range(3):
            ch = color[:,:,i].astype(np.float64)
            mean = ch.mean()
            std = ch.std() + 1e-7
            skew = np.mean(((ch - mean) / std) ** 3)
            features.extend([mean/255, std/255, skew/10])
    
    return np.array(features)


def extract_texture_features(img: np.ndarray) -> np.ndarray:
    """Rich texture features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    # Multi-scale LBP
    for radius in [1, 2, 3]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2), density=True)
        features.extend(hist)
    
    # Gabor filters - captures texture at different orientations
    for theta in np.arange(0, np.pi, np.pi/4):  # 4 orientations
        for freq in [0.1, 0.2, 0.3]:  # 3 frequencies
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 1/freq, 0.5, 0)
            filtered = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kernel)
            features.extend([np.abs(filtered).mean(), filtered.std()])
    
    return np.array(features)


def extract_edge_shape_features(img: np.ndarray) -> np.ndarray:
    """Edge and shape features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    # Edge histogram
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    
    # Grid-based edge density (4x4 grid)
    grid_h, grid_w = h // 4, w // 4
    for i in range(4):
        for j in range(4):
            region = edges[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            features.append(region.mean() / 255.0)
    
    # Sobel gradient statistics
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    
    # Gradient magnitude histogram
    mag_hist, _ = np.histogram(magnitude.ravel(), bins=8, density=True)
    features.extend(mag_hist)
    
    # Gradient direction histogram
    dir_hist, _ = np.histogram(direction.ravel(), bins=8, range=(-np.pi, np.pi), density=True)
    features.extend(dir_hist)
    
    # Hu moments
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    features.extend(hu)
    
    return np.array(features)


def extract_reflectance_features(img: np.ndarray) -> np.ndarray:
    """
    Features specific to material reflectance properties.
    Important for distinguishing glass/metal/plastic.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    features = []
    
    # Brightness/saturation analysis
    v_channel = hsv[:,:,2]
    s_channel = hsv[:,:,1]
    
    # High value regions (specular highlights - common in glass/metal)
    bright_ratio = (v_channel > 200).mean()
    features.append(bright_ratio)
    
    # Low saturation regions (metallic/grey)
    low_sat_ratio = (s_channel < 30).mean()
    features.append(low_sat_ratio)
    
    # Contrast analysis (glass tends to be high contrast)
    contrast = gray.std()
    features.append(contrast / 128.0)
    
    # Local contrast variation
    kernel = np.ones((5,5)) / 25
    local_mean = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    local_var = cv2.filter2D((gray.astype(np.float64) - local_mean)**2, -1, kernel)
    features.append(local_var.mean() / 1000.0)
    features.append(local_var.std() / 1000.0)
    
    # Edge sharpness (materials have different edge characteristics)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.abs(laplacian).mean() / 100.0)
    features.append(laplacian.var() / 10000.0)
    
    # Texture uniformity
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features.append(binary.mean() / 255.0)
    
    return np.array(features)


def extract_spatial_pyramid(img: np.ndarray) -> np.ndarray:
    """Spatial pyramid features for location-aware features."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    
    # Level 0: whole image
    for ch in range(3):
        features.extend([hsv[:,:,ch].mean()/255, hsv[:,:,ch].std()/128])
    
    # Level 1: 2x2 grid
    h, w = hsv.shape[:2]
    h2, w2 = h//2, w//2
    for i in range(2):
        for j in range(2):
            region = hsv[i*h2:(i+1)*h2, j*w2:(j+1)*w2]
            for ch in range(3):
                features.append(region[:,:,ch].mean()/255)
    
    return np.array(features)


def extract_features_svm(img: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive feature vector optimized for material classification.
    """
    features = np.concatenate([
        extract_hog_multiscale(img),        # Shape features
        extract_color_features(img),         # Color distribution
        extract_texture_features(img),       # Texture patterns
        extract_edge_shape_features(img),    # Edge and shape
        extract_reflectance_features(img),   # Material-specific
        extract_spatial_pyramid(img)         # Spatial layout
    ])
    
    return features


if __name__ == "__main__":
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    features = extract_features_svm(test_img)
    print(f"Total feature dimensions: {len(features)}")
