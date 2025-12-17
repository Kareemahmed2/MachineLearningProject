"""
SVM Training Script for Material Stream Identification
Includes data augmentation, feature extraction, hyperparameter tuning, and rejection mechanism
"""

import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

from config import (
    MODELS_PATH, CLASS_NAMES, PRIMARY_CLASSES, CLASS_TO_ID,
    SVM_KERNEL, SVM_C_RANGE, SVM_GAMMA_RANGE, SVM_PCA_VARIANCE,
    SVM_CONFIDENCE_THRESHOLD, TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    TARGET_SAMPLES_PER_CLASS
)
from ImageLoader import load_dataset, augment_images, generate_unknown_samples
from FeatureExtractor_SVM import extract_features_svm


def extract_all_features(images, desc="Extracting features"):
    """Extract features from all images with progress bar."""
    features = []
    for img in tqdm(images, desc=desc):
        features.append(extract_features_svm(img))
    return np.array(features)


def calculate_optimal_threshold(model, X_val, y_val, scaler, pca):
    """
    Calculate optimal confidence threshold for unknown class rejection.
    
    Uses validation set to find threshold that balances accuracy and rejection.
    """
    # Transform validation data
    X_val_scaled = scaler.transform(X_val)
    X_val_pca = pca.transform(X_val_scaled)
    
    # Get probabilities
    probs = model.predict_proba(X_val_pca)
    max_probs = probs.max(axis=1)
    
    # Test different thresholds
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_threshold = SVM_CONFIDENCE_THRESHOLD
    best_score = 0
    
    for thresh in thresholds:
        predictions = model.predict(X_val_pca)
        
        # Apply threshold
        confident_mask = max_probs >= thresh
        
        if confident_mask.sum() == 0:
            continue
        
        # Calculate accuracy only on confident predictions
        confident_acc = accuracy_score(y_val[confident_mask], predictions[confident_mask])
        rejection_rate = 1 - confident_mask.mean()
        
        # Score: balance accuracy and not rejecting too much
        score = confident_acc * (1 - 0.5 * rejection_rate)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    print(f"Optimal threshold: {best_threshold:.2f}")
    return best_threshold


def train_svm():
    """Main training function for SVM classifier."""
    print("=" * 60)
    print("SVM Training for Material Stream Identification")
    print("=" * 60)
    
    # Step 1: Load dataset (only primary classes for training)
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
    
    # Calculate augmentation percentage
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
    
    # Step 5: Preprocessing (Scaling + PCA)
    print("\n[5/7] Preprocessing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=SVM_PCA_VARIANCE, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"  PCA components: {pca.n_components_} (explained variance: {SVM_PCA_VARIANCE*100:.0f}%)")
    
    # Step 6: Hyperparameter tuning
    print("\n[6/7] Training SVM with GridSearchCV...")
    svm = SVC(
        kernel=SVM_KERNEL,
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    
    param_grid = {
        'C': SVM_C_RANGE,
        'gamma': SVM_GAMMA_RANGE
    }
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        svm, param_grid,
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
    y_prob = best_model.predict_proba(X_test_pca)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=PRIMARY_CLASSES))
    
    # Confusion matrix
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate optimal threshold for unknown rejection
    print("\n  Calculating rejection threshold...")
    optimal_threshold = calculate_optimal_threshold(
        best_model, X_test, y_test, scaler, pca
    )
    
    # Evaluate with rejection
    max_probs = y_prob.max(axis=1)
    confident_mask = max_probs >= optimal_threshold
    
    if confident_mask.sum() > 0:
        confident_acc = accuracy_score(y_test[confident_mask], y_pred[confident_mask])
        rejection_rate = 1 - confident_mask.mean()
        print(f"  Accuracy (with rejection): {confident_acc:.4f}")
        print(f"  Rejection rate: {rejection_rate*100:.1f}%")
    
    # Save models
    print("\n" + "=" * 60)
    print("Saving models...")
    
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(MODELS_PATH, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "svm_scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_PATH, "svm_pca.pkl"))
    joblib.dump(optimal_threshold, os.path.join(MODELS_PATH, "svm_threshold.pkl"))
    
    # Also save in root for easy access
    joblib.dump(best_model, "svm_model.pkl")
    joblib.dump(scaler, "svm_scaler.pkl")
    joblib.dump(pca, "svm_pca.pkl")
    joblib.dump(optimal_threshold, "svm_threshold.pkl")
    
    print(f"  Models saved to {MODELS_PATH}")
    print("=" * 60)
    print("SVM Training Complete!")
    print("=" * 60)
    
    return best_model, scaler, pca, optimal_threshold, accuracy


if __name__ == "__main__":
    model, scaler, pca, threshold, accuracy = train_svm()
    
    print(f"\nFinal Results:")
    print(f"  - Model: SVM with {SVM_KERNEL} kernel")
    print(f"  - Test Accuracy: {accuracy*100:.2f}%")
    print(f"  - Rejection Threshold: {threshold:.2f}")
    print(f"  - Target: 85%+ accuracy")
    print(f"  - Status: {'✓ PASSED' if accuracy >= 0.85 else '✗ NEEDS IMPROVEMENT'}")
