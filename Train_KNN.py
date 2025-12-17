import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from ImageLoader import load_images, augment_images
from FeatureExtractor_KNN import extract_features_knn


def load_all():
    data = []
    for label, name in enumerate(["glass","paper","cardboard","plastic","metal","trash"]):
        x, y = load_images(f"./test_set/{name}", label)
        data.append((x, y))

    images = np.concatenate([d[0] for d in data])
    labels = np.concatenate([d[1] for d in data])
    return images, labels


images, labels = load_all()

x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

x_train, y_train = augment_images(x_train, y_train)

X_train = np.array([extract_features_knn(img) for img in x_train])
X_test  = np.array([extract_features_knn(img) for img in x_test])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

knn = KNeighborsClassifier()

param_grid = {
    "n_neighbors": [5, 7, 9],
    "weights": ["distance"],
    "metric": ["cosine"]
}

grid = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

best = grid.best_estimator_

pred = best.predict(X_test)
print("k-NN Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))

joblib.dump(best, "knn_model.pkl")
joblib.dump(scaler, "knn_scaler.pkl")
joblib.dump(pca, "knn_pca.pkl")
