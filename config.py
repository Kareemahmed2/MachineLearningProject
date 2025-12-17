"""
Configuration file for Material Stream Identification System
All constants and hyperparameters are defined here
"""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "test_set")
MODELS_PATH = os.path.join(BASE_DIR, "models")

# Create models directory if it doesn't exist
os.makedirs(MODELS_PATH, exist_ok=True)

# =============================================================================
# IMAGE SETTINGS
# =============================================================================
IMG_SIZE = (128, 128)

# =============================================================================
# CLASS DEFINITIONS
# =============================================================================
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash", "unknown"]
NUM_CLASSES = 7

# Class ID mapping
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Primary classes (for training without unknown)
PRIMARY_CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
NUM_PRIMARY_CLASSES = 6

# =============================================================================
# DATA AUGMENTATION
# =============================================================================
TARGET_SAMPLES_PER_CLASS = 500  # Target balanced dataset size
MIN_AUGMENTATION_FACTOR = 1.3   # Minimum 30% increase as required

# Augmentation parameters
AUG_ROTATION_RANGE = (-15, 15)      # degrees
AUG_BRIGHTNESS_RANGE = (0.8, 1.2)   # multiplier
AUG_CONTRAST_RANGE = (0.8, 1.2)     # multiplier
AUG_BLUR_PROBABILITY = 0.2          # probability of applying blur
AUG_NOISE_PROBABILITY = 0.15        # probability of adding noise

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
# HOG parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP parameters
LBP_RADIUS = [1, 2, 3]
LBP_POINTS_MULTIPLIER = 8

# Color histogram parameters
COLOR_HIST_BINS_LAB = [8, 8, 8]      # For SVM (LAB color space)
COLOR_HIST_BINS_HSV = [16, 8, 8]    # For k-NN (HSV color space)

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# SVM parameters
SVM_KERNEL = "rbf"
SVM_C_RANGE = [1, 5, 10, 20]
SVM_GAMMA_RANGE = ["scale", 0.001, 0.01, 0.1]
SVM_PCA_VARIANCE = 0.95

# k-NN parameters
KNN_K_RANGE = [3, 5, 7, 9, 11]
KNN_WEIGHTS = ["uniform", "distance"]
KNN_METRICS = ["euclidean", "cosine", "manhattan"]
KNN_PCA_VARIANCE = 0.95

# =============================================================================
# REJECTION THRESHOLDS
# =============================================================================
SVM_CONFIDENCE_THRESHOLD = 0.5      # Min probability to accept prediction
KNN_CONFIDENCE_THRESHOLD = 0.4      # Min voting ratio to accept prediction
KNN_DISTANCE_THRESHOLD_PERCENTILE = 95  # Percentile for distance rejection

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# =============================================================================
# DISPLAY COLORS (BGR format for OpenCV)
# =============================================================================
COLORS = {
    "glass": (255, 200, 100),      # Light blue
    "paper": (200, 255, 200),      # Light green
    "cardboard": (100, 180, 255),  # Orange
    "plastic": (255, 100, 255),    # Pink
    "metal": (200, 200, 200),      # Silver
    "trash": (100, 100, 255),      # Red
    "unknown": (0, 255, 255)       # Yellow
}
