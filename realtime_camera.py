import cv2
import joblib
from FeatureExtractor import extract_features

svm = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

IMG_SIZE = (128, 128)

LABELS = {
    0: "glass",
    1: "paper",
    2: "cardboard",
    3: "plastic",
    4: "metal",
    5: "trash",
    6: "unknown"
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, IMG_SIZE)

    features = extract_features(img)
    features = scaler.transform([features])
    features = pca.transform(features)

    probs = svm.predict_proba(features)[0]
    max_prob = probs.max()

    UNKNOWN_THRESHOLD = 0.6

    if max_prob < UNKNOWN_THRESHOLD:
        label = "unknown"
    else:
        label = LABELS[probs.argmax()]

    cv2.putText(
        frame,
        f"Prediction: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Material Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
