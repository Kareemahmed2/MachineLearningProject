"""
k-NN Training Script - Optimized for Higher Accuracy
"""

import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

from config import (
    MODELS_PATH, PRIMARY_CLASSES,
    KNN_PCA_VARIANCE, KNN_CONFIDENCE_THRESHOLD,
    KNN_DISTANCE_THRESHOLD_PERCENTILE,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, TARGET_SAMPLES_PER_CLASS
)
from ImageLoader import load_dataset, augment_images
from FeatureExtractor_KNN import extract_features_knn


def extract_all_features(images, desc="Extracting features"):
    """Extract features with progress bar."""
    features = []
    for img in tqdm(images, desc=desc):
        features.append(extract_features_knn(img))
    return np.array(features)


def calculate_distance_threshold(model, X_train, percentile=95):
    """Calculate distance threshold for rejection."""
    distances, _ = model.kneighbors(X_train)
    avg_distances = distances.mean(axis=1)
    threshold = np.percentile(avg_distances, percentile)
    return threshold


def calculate_voting_confidence(model, X_test):
    """Calculate voting confidence for predictions."""
    distances, indices = model.kneighbors(X_test)
    neighbor_labels = model._y[indices]
    
    confidences = []
    for i in range(len(X_test)):
        labels = neighbor_labels[i]
        pred = model.predict(X_test[i:i+1])[0]
        confidence = np.mean(labels == pred)
        confidences.append(confidence)
    
    return np.array(confidences)


def train_knn():
    """Main k-NN training function."""
    print("=" * 60)
    print("k-NN Training for Material Stream Identification")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n[1/7] Loading dataset...")
    images, labels = load_dataset(include_unknown=False, balance=False)
    
    # Step 2: Split data
    print("\n[2/7] Splitting data...")
    X_train_img, X_test_img, y_train, y_test = train_test_split(
        images, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE
    )
    print(f"  Training images: {len(X_train_img)}")
    print(f"  Test images: {len(X_test_img)}")
    
    # Step 3: Data augmentation
    print("\n[3/7] Applying data augmentation...")
    X_train_img, y_train = augment_images(X_train_img, y_train, TARGET_SAMPLES_PER_CLASS)
    X_train_img = np.array(X_train_img)
    y_train = np.array(y_train)
    
    original_count = len(X_test_img) / TEST_SIZE * (1 - TEST_SIZE)
    aug_percentage = (len(X_train_img) - original_count) / original_count * 100
    print(f"  Augmentation increase: {aug_percentage:.1f}%")
    
    # Step 4: Feature extraction
    print("\n[4/7] Extracting features...")
    start_time = time.time()
    X_train = extract_all_features(X_train_img, "Training features")
    X_test = extract_all_features(X_test_img, "Test features")
    print(f"  Feature extraction time: {time.time() - start_time:.1f}s")
    print(f"  Feature dimensions: {X_train.shape[1]}")
    
    # Step 5: Preprocessing
    print("\n[5/7] Preprocessing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=KNN_PCA_VARIANCE, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"  PCA components: {pca.n_components_} (explained variance: {KNN_PCA_VARIANCE*100:.0f}%)")
    
    # Step 6: Extended hyperparameter search
    print("\n[6/7] Training k-NN with GridSearchCV...")
    knn = KNeighborsClassifier()
    
    # Extended parameter grid
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'cosine', 'manhattan', 'minkowski'],
        'p': [1, 2, 3]  # Power parameter for minkowski
    }
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        knn, param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train_pca, y_train)
    training_time = time.time() - start_time
    
    best_model = grid_search.best_estimator_
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    print(f"  Training time: {training_time:.1f}s")
    
    # Step 7: Evaluation
    print("\n[7/7] Evaluating model...")
    y_pred = best_model.predict(X_test_pca)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=PRIMARY_CLASSES))
    
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Rejection thresholds
    print("\n  Calculating rejection thresholds...")
    distance_threshold = calculate_distance_threshold(
        best_model, X_train_pca, KNN_DISTANCE_THRESHOLD_PERCENTILE
    )
    print(f"  Distance threshold (p{KNN_DISTANCE_THRESHOLD_PERCENTILE}): {distance_threshold:.4f}")
    
    confidences = calculate_voting_confidence(best_model, X_test_pca)
    distances, _ = best_model.kneighbors(X_test_pca)
    avg_distances = distances.mean(axis=1)
    
    confident_mask = (avg_distances <= distance_threshold) & (confidences >= KNN_CONFIDENCE_THRESHOLD)
    
    if confident_mask.sum() > 0:
        confident_acc = accuracy_score(y_test[confident_mask], y_pred[confident_mask])
        rejection_rate = 1 - confident_mask.mean()
        print(f"  Accuracy (with rejection): {confident_acc:.4f}")
        print(f"  Rejection rate: {rejection_rate*100:.1f}%")
    
    # Save models
    print("\n" + "=" * 60)
    print("Saving models...")
    
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    threshold_config = {
        'distance_threshold': distance_threshold,
        'confidence_threshold': KNN_CONFIDENCE_THRESHOLD
    }
    
    joblib.dump(best_model, os.path.join(MODELS_PATH, "knn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "knn_scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_PATH, "knn_pca.pkl"))
    joblib.dump(threshold_config, os.path.join(MODELS_PATH, "knn_threshold.pkl"))
    
    joblib.dump(best_model, "knn_model.pkl")
    joblib.dump(scaler, "knn_scaler.pkl")
    joblib.dump(pca, "knn_pca.pkl")
    joblib.dump(threshold_config, "knn_threshold.pkl")
    
    print(f"  Models saved to {MODELS_PATH}")
    print("=" * 60)
    print("k-NN Training Complete!")
    print("=" * 60)
    
    return best_model, scaler, pca, threshold_config, accuracy


if __name__ == "__main__":
    model, scaler, pca, thresholds, accuracy = train_knn()
    
    print(f"\nFinal Results:")
    print(f"  - Model: k-NN with {model.n_neighbors} neighbors")
    print(f"  - Weights: {model.weights}")
    print(f"  - Metric: {model.metric}")
    print(f"  - Test Accuracy: {accuracy*100:.2f}%")
    print(f"  - Distance Threshold: {thresholds['distance_threshold']:.4f}")
    print(f"  - Target: 85%+ accuracy")
    print(f"  - Status: {'✓ PASSED' if accuracy >= 0.85 else '✗ NEEDS IMPROVEMENT'}")
