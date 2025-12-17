"""
SVM Training - Optimized Version 3
With extended hyperparameter search and better regularization
"""

import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

from config import (
    MODELS_PATH, PRIMARY_CLASSES,
    SVM_KERNEL, SVM_PCA_VARIANCE,
    SVM_CONFIDENCE_THRESHOLD, TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    TARGET_SAMPLES_PER_CLASS
)
from ImageLoader import load_dataset, augment_images
from FeatureExtractor_SVM import extract_features_svm


def extract_all_features(images, desc="Extracting features"):
    features = []
    for img in tqdm(images, desc=desc):
        features.append(extract_features_svm(img))
    return np.array(features)


def train_svm():
    print("=" * 60)
    print("SVM Training v3 - Material Stream Identification")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/7] Loading dataset...")
    images, labels = load_dataset(include_unknown=False)
    
    # Split
    print("\n[2/7] Splitting data...")
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    print(f"  Training: {len(X_train_img)}, Test: {len(X_test_img)}")
    
    # Augmentation
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
    
    # Preprocessing
    print("\n[5/7] Preprocessing...")
    scaler = RobustScaler()  # More robust to outliers
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    pca = PCA(n_components=0.98, random_state=RANDOM_STATE)  # Keep more variance
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    print(f"  PCA: {pca.n_components_} components (98% variance)")
    
    # Training with extensive grid search
    print("\n[6/7] Training SVM...")
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 500],
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly'],
        'degree': [2, 3]  # For poly kernel
    }
    
    svm = SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE, cache_size=1000)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    
    t0 = time.time()
    grid.fit(X_train_p, y_train)
    print(f"  Best params: {grid.best_params_}")
    print(f"  CV score: {grid.best_score_:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    best = grid.best_estimator_
    
    # Evaluation
    print("\n[7/7] Evaluating...")
    y_pred = best.predict(X_test_p)
    y_prob = best.predict_proba(X_test_p)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=PRIMARY_CLASSES))
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Threshold
    thresholds = np.arange(0.3, 0.7, 0.05)
    max_probs = y_prob.max(axis=1)
    
    best_thresh, best_acc_rej = 0.5, acc
    for t in thresholds:
        mask = max_probs >= t
        if mask.sum() > 0:
            acc_t = accuracy_score(y_test[mask], y_pred[mask])
            if acc_t > best_acc_rej:
                best_acc_rej = acc_t
                best_thresh = t
    
    print(f"\n  Rejection threshold: {best_thresh:.2f}")
    mask = max_probs >= best_thresh
    if mask.sum() > 0:
        print(f"  Accuracy with rejection: {accuracy_score(y_test[mask], y_pred[mask]):.4f}")
        print(f"  Rejection rate: {(1-mask.mean())*100:.1f}%")
    
    # Save
    print("\n" + "=" * 60)
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    for path_prefix in [os.path.join(MODELS_PATH, "svm"), "svm"]:
        joblib.dump(best, f"{path_prefix}_model.pkl")
        joblib.dump(scaler, f"{path_prefix}_scaler.pkl")
        joblib.dump(pca, f"{path_prefix}_pca.pkl")
        joblib.dump(best_thresh, f"{path_prefix}_threshold.pkl")
    
    print("Models saved!")
    print("=" * 60)
    
    return best, scaler, pca, best_thresh, acc


if __name__ == "__main__":
    model, scaler, pca, thresh, acc = train_svm()
    print(f"\nFinal: {acc*100:.2f}% - {'✓ PASSED' if acc >= 0.85 else '✗ NEEDS IMPROVEMENT'}")
