import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from ImageLoader import load_images, balance_dataset
from FeatureExtractor_KNN import extract_features_knn


def load_all_data(base_path="./test_set"):

    data = []
    class_names = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

    for label, name in enumerate(class_names):
        folder = f"{base_path}/{name}"
        print(f"Loading {name} from {folder}...")
        x, y = load_images(folder, label)
        print(f"  Loaded {len(x)} images")
        data.append((x, y))

    images = np.concatenate([d[0] for d in data])
    labels = np.concatenate([d[1] for d in data])

    print(f"\nTotal images loaded: {len(images)}")
    return images, labels


print("=" * 60)
print("LOADING DATA")
print("=" * 60)
images, labels = load_all_data()

# Split data
print("\nSplitting data (80% train, 20% test)...")
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

# Balance and augment
print("\n" + "=" * 60)
print("AUGMENTING TRAINING DATA TO 500 PER CLASS")
print("=" * 60)
x_train, y_train = balance_dataset(x_train, y_train, target_count=500)

print(f"After augmentation: {len(x_train)} training images")
unique, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} samples")

# Extract features
print("\n" + "=" * 60)
print("EXTRACTING FEATURES")
print("=" * 60)
print("Extracting training features...")
X_train = np.array([extract_features_knn(img) for img in x_train])
print("Extracting test features...")
X_test = np.array([extract_features_knn(img) for img in x_test])

print(f"Feature vector size: {X_train.shape[1]}")

# Normalize features (critical for k-NN!)
print("\nNormalizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA (optional but helps with distance computation)
print("Applying PCA (95% variance)...")
pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(f"PCA reduced features to: {X_train.shape[1]} dimensions")

# Train k-NN with comprehensive grid search
print("\n" + "=" * 60)
print("TRAINING k-NN WITH GRID SEARCH")
print("=" * 60)

knn = KNeighborsClassifier()

# Expanded parameter grid for k-NN
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "cosine"],
    "p": [1, 2]  # For minkowski distance
}

print("Grid search parameters:", param_grid)
print("Using 5-fold cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    knn,
    param_grid,
    cv=cv,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

print("\n" + "=" * 60)
print("GRID SEARCH RESULTS")
print("=" * 60)
print(f"Best parameters: {grid.best_params_}")
print(f"Best cross-validation score: {grid.best_score_:.4f}")

best_knn = grid.best_estimator_

# Evaluate on test set
print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)
pred = best_knn.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred,
                            target_names=["glass", "paper", "cardboard",
                                          "plastic", "metal", "trash"]))

# Save models
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)
joblib.dump(best_knn, "knn_model.pkl")
joblib.dump(scaler, "knn_scaler.pkl")
joblib.dump(pca, "knn_pca.pkl")

print("✓ Saved: knn_model.pkl")
print("✓ Saved: knn_scaler.pkl")
print("✓ Saved: knn_pca.pkl")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Final Test Accuracy: {acc * 100:.2f}%")

if acc >= 0.85:
    print("✓ Target accuracy of 85% achieved!")
else:
    print(f"⚠ Need {(0.85 - acc) * 100:.2f}% more to reach 85%")