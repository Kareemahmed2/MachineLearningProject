"""
k-NN Training - Optimized Version 3
With extensive hyperparameter tuning
"""

import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

from config import (
    MODELS_PATH, PRIMARY_CLASSES,
    KNN_PCA_VARIANCE, KNN_CONFIDENCE_THRESHOLD,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, TARGET_SAMPLES_PER_CLASS
)
from ImageLoader import load_dataset, augment_images
from FeatureExtractor_KNN import extract_features_knn


def extract_all_features(images, desc="Extracting"):
    features = []
    for img in tqdm(images, desc=desc):
        features.append(extract_features_knn(img))
    return np.array(features)


def train_knn():
    print("=" * 60)
    print("k-NN Training v3 - Material Stream Identification")
    print("=" * 60)
    
    # Load
    print("\n[1/7] Loading dataset...")
    images, labels = load_dataset(include_unknown=False)
    
    # Split
    print("\n[2/7] Splitting...")
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    print(f"  Training: {len(X_train_img)}, Test: {len(X_test_img)}")
    
    # Augment
    print("\n[3/7] Augmenting...")
    X_train_img, y_train = augment_images(X_train_img, y_train, TARGET_SAMPLES_PER_CLASS)
    X_train_img = np.array(X_train_img)
    y_train = np.array(y_train)
    
    # Features
    print("\n[4/7] Extracting features...")
    t0 = time.time()
    X_train = extract_all_features(X_train_img, "Training")
    X_test = extract_all_features(X_test_img, "Test")
    print(f"  Time: {time.time()-t0:.1f}s, Dims: {X_train.shape[1]}")
    
    # Preprocess
    print("\n[5/7] Preprocessing...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    pca = PCA(n_components=0.98, random_state=RANDOM_STATE)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    print(f"  PCA: {pca.n_components_} components")
    
    # Training
    print("\n[6/7] Training k-NN...")
    
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'cosine', 'chebyshev'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    }
    
    knn = KNeighborsClassifier()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    
    t0 = time.time()
    grid.fit(X_train_p, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  CV score: {grid.best_score_:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    best = grid.best_estimator_
    
    # Evaluate
    print("\n[7/7] Evaluating...")
    y_pred = best.predict(X_test_p)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=PRIMARY_CLASSES))
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Rejection thresholds
    distances, indices = best.kneighbors(X_test_p)
    avg_dist = distances.mean(axis=1)
    
    # Calculate voting confidence
    neighbor_labels = best._y[indices]
    confidences = []
    for i in range(len(X_test_p)):
        conf = (neighbor_labels[i] == y_pred[i]).mean()
        confidences.append(conf)
    confidences = np.array(confidences)
    
    # Find best thresholds
    dist_thresh = np.percentile(avg_dist, 95)
    
    mask = (avg_dist <= dist_thresh) & (confidences >= 0.4)
    if mask.sum() > 0:
        print(f"\n  Distance threshold: {dist_thresh:.4f}")
        print(f"  Accuracy with rejection: {accuracy_score(y_test[mask], y_pred[mask]):.4f}")
        print(f"  Rejection rate: {(1-mask.mean())*100:.1f}%")
    
    thresh_config = {
        'distance_threshold': dist_thresh,
        'confidence_threshold': 0.4
    }
    
    # Save
    print("\n" + "=" * 60)
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    for path_prefix in [os.path.join(MODELS_PATH, "knn"), "knn"]:
        joblib.dump(best, f"{path_prefix}_model.pkl")
        joblib.dump(scaler, f"{path_prefix}_scaler.pkl")
        joblib.dump(pca, f"{path_prefix}_pca.pkl")
        joblib.dump(thresh_config, f"{path_prefix}_threshold.pkl")
    
    print("Models saved!")
    print("=" * 60)
    
    return best, scaler, pca, thresh_config, acc


if __name__ == "__main__":
    model, scaler, pca, thresh, acc = train_knn()
    print(f"\nFinal: {acc*100:.2f}% - {'✓ PASSED' if acc >= 0.85 else '✗ NEEDS IMPROVEMENT'}")
