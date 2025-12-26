import cv2
import joblib
import numpy as np
import tensorflow as tf

from FeatureExtractor_KNN import extract_features_knn

# ===================== CONSTANTS =====================
IMG_SIZE = (128, 128)

LABELS = {
    0: "glass",
    1: "paper",
    2: "cardboard",
    3: "plastic",
    4: "metal",
    5: "trash"
}

UNKNOWN_DISTANCE_THRESHOLD = 0.75


# ===================== MODEL LOADER =====================
def load_model(model_type):
    if model_type == "knn":
        return {
            "model": joblib.load("knn_model.pkl"),
            "scaler": joblib.load("knn_scaler.pkl"),
            "pca": joblib.load("knn_pca.pkl")
        }

    if model_type == "svm":
        return {
            "model": joblib.load("svm_model.pkl"),
            "scaler": joblib.load("svm_scaler.pkl"),
            "pca": joblib.load("svm_pca.pkl")
        }

    if model_type == "cnn":
        return tf.keras.models.load_model("cnn_model.keras")

    if model_type == "mobilenet":
        return tf.keras.models.load_model("cnn_mobilenet.keras")

    raise ValueError("Invalid model type")


# ===================== PREDICTION =====================
def predict(frame, model_type, artifacts):
    img = cv2.resize(frame, IMG_SIZE)

    # -------- KNN --------
    if model_type == "knn":
        features = extract_features_knn(img)
        features = artifacts["scaler"].transform([features])
        features = artifacts["pca"].transform(features)

        distances, _ = artifacts["model"].kneighbors(features, n_neighbors=1)
        min_dist = distances[0][0]

        if min_dist > UNKNOWN_DISTANCE_THRESHOLD:
            return "unknown", min_dist
        else:
            pred = artifacts["model"].predict(features)[0]
            return LABELS[pred], min_dist

    # -------- SVM --------
    if model_type == "svm":
        features = extract_features_knn(img)
        features = artifacts["scaler"].transform([features])
        features = artifacts["pca"].transform(features)

        pred = artifacts["model"].predict(features)[0]
        return LABELS[pred], None

    # -------- CNN --------
    if model_type == "cnn":
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        preds = artifacts.predict(img, verbose=0)
        pred = np.argmax(preds)
        return LABELS[pred], np.max(preds)

    # -------- MobileNet --------
    if model_type == "mobilenet":
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        preds = artifacts.predict(img, verbose=0)
        pred = np.argmax(preds)
        return LABELS[pred], np.max(preds)


# ===================== CAMERA =====================
def run_camera(model_type, artifacts):
    cap = cv2.VideoCapture(0)
    print(f"\n📷 Running {model_type.upper()} — press 'q' to stop camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, score = predict(frame, model_type, artifacts)

        color = (0, 255, 0) if label != "unknown" else (0, 0, 255)

        cv2.putText(
            frame,
            f"Prediction: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        if score is not None:
            cv2.putText(
                frame,
                f"Score: {score:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("Material Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===================== MAIN MENU =====================
while True:
    print("\nSelect model:")
    print("1 - KNN")
    print("2 - SVM")
    print("3 - CNN")
    print("4 - MobileNet")
    print("q - Quit")

    choice = input("Your choice: ").strip().lower()

    if choice == "q":
        print("👋 Exiting program")
        break

    model_map = {
        "1": "knn",
        "2": "svm",
        "3": "cnn",
        "4": "mobilenet"
    }

    if choice not in model_map:
        print("❌ Invalid choice")
        continue

    model_type = model_map[choice]

    print(f"🔄 Loading {model_type.upper()} model...")
    artifacts = load_model(model_type)

    run_camera(model_type, artifacts)
