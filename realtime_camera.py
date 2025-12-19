import cv2
import joblib
import numpy as np
from FeatureExtractor_KNN import extract_features_knn

# Load trained KNN artifacts
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("knn_scaler.pkl")
pca = joblib.load("knn_pca.pkl")

IMG_SIZE = (128, 128)

LABELS = {
    0: "glass",
    1: "paper",
    2: "cardboard",
    3: "plastic",
    4: "metal",
    5: "trash"
}

# Distance threshold for UNKNOWN detection
UNKNOWN_DISTANCE_THRESHOLD = 0.75  # tune if needed

cap = cv2.VideoCapture(0)

print("📷 KNN Realtime Classification Started (press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match training
    img = cv2.resize(frame, IMG_SIZE)

    # --- Feature extraction pipeline (MUST match training) ---
    features = extract_features_knn(img)
    features = scaler.transform([features])
    features = pca.transform(features)

    # --- KNN distance-based prediction ---
    distances, indices = knn.kneighbors(features, n_neighbors=1)
    min_dist = distances[0][0]

    if min_dist > UNKNOWN_DISTANCE_THRESHOLD:
        label = "unknown"
        color = (0, 0, 255)  # red
    else:
        pred = knn.predict(features)[0]
        label = LABELS[pred]
        color = (0, 255, 0)  # green

    # Display result
    cv2.putText(
        frame,
        f"Prediction: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Dist: {min_dist:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.imshow("Material Classification (KNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
