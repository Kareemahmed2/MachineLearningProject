# Material Stream Identification System
## Technical Report

**Course:** Machine Learning  
**Faculty:** Faculty of Computing and Artificial Intelligence, Cairo University  
**Team Members:** [Add your names and IDs here]  
**TA Name:** [Add TA name here]  
**Date:** December 2025

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Description](#2-dataset-description)
3. [Data Augmentation](#3-data-augmentation)
4. [Feature Extraction](#4-feature-extraction)
5. [Classifier Implementation](#5-classifier-implementation)
6. [Rejection Mechanism](#6-rejection-mechanism)
7. [Experimental Results](#7-experimental-results)
8. [Classifier Comparison](#8-classifier-comparison)
9. [Real-Time System](#9-real-time-system)
10. [Conclusions](#10-conclusions)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 Problem Statement

The efficient and automated sorting of post-consumer waste is a critical bottleneck in achieving circular economy goals. This project implements an **Automated Material Stream Identification (MSI) System** using fundamental Machine Learning techniques to classify waste materials into seven distinct categories.

### 1.2 Objectives

1. Implement a complete ML pipeline: Data Preprocessing → Feature Extraction → Classification → Evaluation
2. Develop two classifiers: Support Vector Machine (SVM) and k-Nearest Neighbors (k-NN)
3. Achieve minimum 85% validation accuracy across primary classes
4. Deploy a real-time classification system using live camera feed

### 1.3 Material Classes

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | Glass | Amorphous solid materials (bottles, jars) |
| 1 | Paper | Thin cellulose materials (newspapers, office paper) |
| 2 | Cardboard | Multi-layer cellulose fiber structures |
| 3 | Plastic | High-molecular-weight organic compounds |
| 4 | Metal | Elemental or compound metallic substances |
| 5 | Trash | Non-recyclable or contaminated waste |
| 6 | Unknown | Out-of-distribution or uncertain items |

---

## 2. Dataset Description

### 2.1 Original Dataset Statistics

| Class | Original Count | Percentage |
|-------|----------------|------------|
| Glass | 401 | 20.4% |
| Paper | 476 | 24.2% |
| Cardboard | 259 | 13.2% |
| Plastic | 386 | 19.6% |
| Metal | 328 | 16.7% |
| Trash | 110 | 5.6% |
| **Total** | **1960** | **100%** |

### 2.2 Dataset Characteristics

- **Image Format:** JPEG
- **Image Size:** Variable (resized to 128×128 for processing)
- **Color Space:** BGR (OpenCV default)
- **Class Imbalance:** Significant (Trash has only 110 samples vs Paper with 476)

### 2.3 Train/Test Split

- **Split Ratio:** 80% Training, 20% Testing
- **Stratification:** Yes (maintains class proportions)
- **Random Seed:** 42 (for reproducibility)

---

## 3. Data Augmentation

### 3.1 Motivation

Data augmentation is mandatory to:
1. Increase training sample size by minimum 30%
2. Balance class distribution (target: ~500 samples per class)
3. Improve model generalization
4. Handle variations in lighting, orientation, and scale

### 3.2 Augmentation Techniques

| Technique | Parameters | Justification |
|-----------|------------|---------------|
| **Horizontal Flip** | 50% probability | Materials can appear from any angle |
| **Vertical Flip** | 50% probability | Adds orientation invariance |
| **Random Rotation** | ±15 degrees | Simulates camera angle variations |
| **Brightness Adjustment** | 0.8-1.2 multiplier | Handles lighting variations |
| **Contrast Adjustment** | 0.8-1.2 multiplier | Compensates for camera quality |
| **Gaussian Blur** | Kernel 3-5, 20% prob | Simulates out-of-focus images |
| **Gaussian Noise** | σ=10, 15% prob | Adds robustness to sensor noise |
| **Combined Transform** | Multiple above | Creates diverse variations |

### 3.3 Augmentation Results

| Class | Before | After | Increase |
|-------|--------|-------|----------|
| Glass | 321 | 500 | +55.8% |
| Paper | 381 | 500 | +31.2% |
| Cardboard | 207 | 500 | +141.5% |
| Plastic | 309 | 500 | +61.8% |
| Metal | 262 | 500 | +90.8% |
| Trash | 88 | 500 | +468.2% |
| **Total** | **1568** | **3000** | **+91.3%** |

> **Note:** Augmentation increase exceeds the required 30% minimum significantly to ensure class balance.

---

## 4. Feature Extraction

### 4.1 Overview

Feature extraction converts raw 2D/3D image data into 1D numerical feature vectors suitable for classification. Different feature types capture different aspects of the image:

- **Shape features:** Capture object contours and edges
- **Texture features:** Capture surface patterns
- **Color features:** Capture material-specific color distributions

### 4.2 SVM Feature Extraction

The SVM classifier uses a comprehensive feature vector (~2300 dimensions):

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| **HOG (Histogram of Oriented Gradients)** | ~1764 | Captures edge orientations and shape information. Parameters: 9 orientations, 8×8 pixels/cell, 2×2 cells/block |
| **Color Histogram (LAB)** | 512 | 3D histogram in LAB color space (8×8×8 bins). LAB is perceptually uniform, better for material differentiation |
| **Multi-scale LBP** | 37 | Local Binary Patterns at radii 1, 2, 3. Captures micro-texture patterns |
| **GLCM Features** | 32 | Gray-Level Co-occurrence Matrix: contrast, homogeneity, energy, correlation |
| **Entropy** | 4 | Local entropy statistics (mean, std, min, max) |
| **Edge Features** | 4 | Canny edge density and Sobel gradient statistics |
| **Hu Moments** | 7 | Shape descriptors invariant to translation, scale, rotation |
| **Color Moments** | 27 | Mean, std, skewness per channel (BGR, HSV, LAB) |

**Justification for SVM Features:**
- SVMs work well with high-dimensional data
- Comprehensive features allow the kernel to find complex decision boundaries
- PCA reduces dimensionality while preserving 95% variance

### 4.3 k-NN Feature Extraction

The k-NN classifier uses a compact feature vector (~1090 dimensions):

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| **Color Histogram (HSV)** | 1024 | 3D histogram in HSV space (16×8×8 bins). HSV is robust to lighting |
| **LBP** | 18 | Uniform LBP with 16 points, radius 2 |
| **Intensity Statistics** | 6 | Mean and std per BGR channel |
| **Edge Distribution** | 8 | Quadrant-wise edge density + global statistics |
| **Gabor Texture** | 16 | Gabor filter responses at 2 frequencies, 4 orientations |
| **Shape Features** | 6 | Area, circularity, aspect ratio, extent, solidity, perimeter |
| **Dominant Colors** | 12 | Top 3 dominant colors via k-means + proportions |

**Justification for k-NN Features:**
- k-NN performance degrades in very high dimensions (curse of dimensionality)
- Compact, normalized features improve distance metric reliability
- Each feature is individually meaningful for material properties

### 4.4 Feature Preprocessing

1. **Standardization:** StandardScaler (zero mean, unit variance)
2. **Dimensionality Reduction:** PCA retaining 95% variance
3. **Final Dimensions:**
   - SVM: ~2300 → ~150-200 components
   - k-NN: ~1090 → ~100-150 components

---

## 5. Classifier Implementation

### 5.1 Support Vector Machine (SVM)

#### Architecture

```
Input Features (n dimensions)
        ↓
StandardScaler (normalization)
        ↓
PCA (95% variance)
        ↓
SVM with RBF Kernel
        ↓
Probability Calibration (Platt Scaling)
        ↓
Output: Class Probabilities
```

#### Hyperparameters

| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| **Kernel** | RBF (Radial Basis Function) | Suitable for non-linear boundaries |
| **C** | [Best from GridSearch] | Regularization parameter |
| **Gamma** | [Best from GridSearch] | Kernel coefficient |
| **class_weight** | 'balanced' | Handles class imbalance |

**Kernel Choice Justification:**
- RBF kernel is the most versatile and works well when the relationship between features and classes is non-linear
- Unlike linear kernel, RBF can model complex decision boundaries
- Unlike polynomial kernel, RBF has fewer hyperparameters to tune

#### Training Process

1. GridSearchCV with 5-fold stratified cross-validation
2. Parameter grid: C ∈ {1, 5, 10, 20}, γ ∈ {scale, 0.001, 0.01, 0.1}
3. Scoring metric: Accuracy
4. Enable probability=True for confidence estimation

### 5.2 k-Nearest Neighbors (k-NN)

#### Architecture

```
Input Features (n dimensions)
        ↓
StandardScaler (normalization)
        ↓
PCA (95% variance)
        ↓
k-NN Distance Calculation
        ↓
Weighted Voting
        ↓
Output: Class + Distance Statistics
```

#### Hyperparameters

| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| **k (n_neighbors)** | [Best from GridSearch] | Usually 5-11 |
| **weights** | 'distance' | Closer neighbors have more influence |
| **metric** | [Best from GridSearch] | euclidean, cosine, or manhattan |

**Weight Scheme Justification:**
- Distance-weighted voting gives more influence to closer (more similar) neighbors
- Reduces the impact of outliers in the neighbor set
- More reliable than uniform voting for material classification

#### Training Process

1. GridSearchCV with 5-fold stratified cross-validation
2. Parameter grid: k ∈ {3, 5, 7, 9, 11}, weights ∈ {uniform, distance}, metric ∈ {euclidean, cosine, manhattan}
3. Store training data for distance threshold calculation

---

## 6. Rejection Mechanism

### 6.1 Purpose

The rejection mechanism classifies uncertain predictions as "Unknown" (class 6). This is critical for:
- Handling out-of-distribution items
- Avoiding confident misclassifications
- Improving system reliability in real-world deployment

### 6.2 SVM Rejection (Probability-Based)

**Mechanism:**
```python
probabilities = svm.predict_proba(features)
max_prob = probabilities.max()

if max_prob < threshold:
    prediction = "Unknown"
```

**Threshold Selection:**
- Optimal threshold calculated on validation set
- Balances accuracy on confident predictions vs rejection rate
- Typical range: 0.4-0.6

### 6.3 k-NN Rejection (Distance + Voting-Based)

**Mechanism:**
```python
# Distance-based
distances, _ = knn.kneighbors(features)
avg_distance = distances.mean()

# Voting-based
neighbor_labels = knn._y[indices]
voting_confidence = (neighbor_labels == prediction).mean()

if avg_distance > distance_threshold or voting_confidence < voting_threshold:
    prediction = "Unknown"
```

**Threshold Selection:**
- Distance threshold: 95th percentile of training set distances
- Voting threshold: typically 0.4 (40% of neighbors agree)

### 6.4 Comparison of Rejection Methods

| Aspect | SVM (Probability) | k-NN (Distance + Voting) |
|--------|-------------------|--------------------------|
| Basis | Posterior probabilities | Geometric distance |
| Calibration | Requires Platt scaling | Inherently interpretable |
| Speed | Fast (single forward pass) | Slower (distance computation) |
| Sensitivity | Good for borderline cases | Good for outliers |

---

## 7. Experimental Results

### 7.1 SVM Results

| Metric | Value |
|--------|-------|
| Test Accuracy | [XX.XX%] |
| CV Score | [XX.XX%] |
| Best C | [value] |
| Best γ | [value] |
| PCA Components | [XX] |
| Rejection Rate | [XX.XX%] |
| Accuracy (with rejection) | [XX.XX%] |

#### Confusion Matrix (SVM)

```
[Insert confusion matrix here]
```

#### Per-Class Performance (SVM)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glass | | | |
| Paper | | | |
| Cardboard | | | |
| Plastic | | | |
| Metal | | | |
| Trash | | | |

### 7.2 k-NN Results

| Metric | Value |
|--------|-------|
| Test Accuracy | [XX.XX%] |
| CV Score | [XX.XX%] |
| Best k | [value] |
| Best metric | [value] |
| PCA Components | [XX] |
| Rejection Rate | [XX.XX%] |
| Accuracy (with rejection) | [XX.XX%] |

#### Confusion Matrix (k-NN)

```
[Insert confusion matrix here]
```

#### Per-Class Performance (k-NN)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glass | | | |
| Paper | | | |
| Cardboard | | | |
| Plastic | | | |
| Metal | | | |
| Trash | | | |

---

## 8. Classifier Comparison

### 8.1 Performance Comparison

| Metric | SVM | k-NN | Winner |
|--------|-----|------|--------|
| Test Accuracy | | | |
| Training Time | | | |
| Prediction Time | | | |
| Memory Usage | | | |
| Rejection Quality | | | |

### 8.2 Feature Extraction Comparison

| Aspect | SVM Features | k-NN Features |
|--------|--------------|---------------|
| Dimensions | ~2300 | ~1090 |
| Extraction Time | Slower | Faster |
| Information Richness | Higher | Moderate |
| Distance Metric Suitability | N/A | Better |

### 8.3 Trade-offs Analysis

#### SVM Advantages:
1. Generally higher accuracy with complex decision boundaries
2. Better handling of high-dimensional features
3. More robust to overfitting with proper regularization
4. Probability estimates enable soft rejection

#### SVM Disadvantages:
1. Slower prediction for large support vector sets
2. Requires careful kernel selection
3. Less interpretable predictions

#### k-NN Advantages:
1. Simple and intuitive algorithm
2. No training phase (lazy learning)
3. Naturally handles multi-class problems
4. Distance-based rejection is interpretable

#### k-NN Disadvantages:
1. Slow prediction (computes all distances)
2. Sensitive to curse of dimensionality
3. Memory-intensive (stores all training data)
4. Requires good distance metric selection

### 8.4 Recommendation

Based on our experiments, **[SVM/k-NN]** is recommended for the final deployment due to:
1. [Reason 1]
2. [Reason 2]
3. [Reason 3]

---

## 9. Real-Time System

### 9.1 System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Camera    │ ──▶ │   Capture    │ ──▶ │   Resize    │
│   (USB)     │     │   Frame      │     │  (128×128)  │
└─────────────┘     └──────────────┘     └─────────────┘
                                                ↓
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Display   │ ◀── │   Draw UI    │ ◀── │  Classify   │
│   Result    │     │   Overlay    │     │   (Model)   │
└─────────────┘     └──────────────┘     └─────────────┘
```

### 9.2 Features

| Feature | Description |
|---------|-------------|
| **Model Selection** | Switch between SVM and k-NN with 'm' key |
| **Confidence Display** | Visual progress bar showing prediction confidence |
| **Color Coding** | Different colors for each class |
| **Temporal Smoothing** | Averages predictions over 5 frames |
| **FPS Counter** | Real-time performance monitoring |
| **Screenshot** | Save current frame with 's' key |

### 9.3 Performance

| Metric | Value |
|--------|-------|
| Frame Rate | [XX] FPS |
| Prediction Latency | [XX] ms |
| Memory Usage | [XX] MB |

---

## 10. Conclusions

### 10.1 Achievements

1. ✓ Implemented complete ML pipeline with data preprocessing, feature extraction, and classification
2. ✓ Developed two classifiers (SVM and k-NN) with proper hyperparameter tuning
3. ✓ Achieved [XX]% validation accuracy (target: 85%)
4. ✓ Implemented rejection mechanism for handling unknown/uncertain inputs
5. ✓ Deployed real-time classification system with webcam support

### 10.2 Challenges Faced

1. **Class Imbalance:** Addressed through data augmentation and balanced class weights
2. **Feature Dimensionality:** Managed with PCA while preserving 95% variance
3. **Unknown Class:** Generated synthetic unknown samples and implemented rejection

### 10.3 Future Work

1. Explore deep learning features (CNN-based feature extraction)
2. Implement ensemble methods combining SVM and k-NN
3. Add more material classes
4. Improve real-time performance with GPU acceleration

---

## 11. References

1. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. CVPR.
2. Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI.
3. Haralick, R. M., Shanmugam, K., & Dinstein, I. H. (1973). Textural features for image classification. IEEE Transactions on systems, man, and cybernetics.
4. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning.
5. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE transactions on information theory.

---

## Appendix A: Code Structure

```
MachineLearningProject/
├── config.py                  # Configuration and constants
├── ImageLoader.py             # Data loading and augmentation
├── FeatureExtractor_SVM.py    # SVM feature extraction
├── FeatureExtractor_KNN.py    # k-NN feature extraction
├── Train_SVM.py               # SVM training script
├── Train_KNN.py               # k-NN training script
├── realtime_camera.py         # Real-time application
├── requirements.txt           # Dependencies
├── test_set/                  # Dataset folder
│   ├── glass/
│   ├── paper/
│   ├── cardboard/
│   ├── plastic/
│   ├── metal/
│   └── trash/
└── models/                    # Saved models
    ├── svm_model.pkl
    ├── svm_scaler.pkl
    ├── svm_pca.pkl
    ├── knn_model.pkl
    ├── knn_scaler.pkl
    └── knn_pca.pkl
```

## Appendix B: How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train SVM model
python Train_SVM.py

# Train k-NN model
python Train_KNN.py

# Run real-time classification
python realtime_camera.py --model svm
python realtime_camera.py --model knn
```
