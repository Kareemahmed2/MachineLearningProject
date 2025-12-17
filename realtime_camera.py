"""
Real-Time Material Classification Application
Processes live camera frames and displays classification results
Supports both SVM and k-NN models with rejection mechanism
"""

import cv2
import numpy as np
import joblib
import argparse
import time
import os

from config import (
    IMG_SIZE, CLASS_NAMES, ID_TO_CLASS, COLORS, MODELS_PATH,
    SVM_CONFIDENCE_THRESHOLD, KNN_CONFIDENCE_THRESHOLD
)
from FeatureExtractor_SVM import extract_features_svm
from FeatureExtractor_KNN import extract_features_knn


class MaterialClassifier:
    """Unified classifier interface for SVM and k-NN models."""
    
    def __init__(self, model_type='svm'):
        """
        Initialize classifier with specified model type.
        
        Args:
            model_type: 'svm' or 'knn'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.pca = None
        self.threshold = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and preprocessing components."""
        prefix = self.model_type
        
        # Try loading from models folder first, then root
        model_paths = [
            (os.path.join(MODELS_PATH, f"{prefix}_model.pkl"),
             os.path.join(MODELS_PATH, f"{prefix}_scaler.pkl"),
             os.path.join(MODELS_PATH, f"{prefix}_pca.pkl"),
             os.path.join(MODELS_PATH, f"{prefix}_threshold.pkl")),
            (f"{prefix}_model.pkl",
             f"{prefix}_scaler.pkl",
             f"{prefix}_pca.pkl",
             f"{prefix}_threshold.pkl")
        ]
        
        loaded = False
        for model_path, scaler_path, pca_path, threshold_path in model_paths:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.pca = joblib.load(pca_path)
                
                if os.path.exists(threshold_path):
                    self.threshold = joblib.load(threshold_path)
                else:
                    # Default thresholds
                    if self.model_type == 'svm':
                        self.threshold = SVM_CONFIDENCE_THRESHOLD
                    else:
                        self.threshold = {
                            'distance_threshold': 1.0,
                            'confidence_threshold': KNN_CONFIDENCE_THRESHOLD
                        }
                
                loaded = True
                print(f"Loaded {self.model_type.upper()} model from {model_path}")
                break
        
        if not loaded:
            raise FileNotFoundError(
                f"Could not find {self.model_type.upper()} model. "
                f"Please run Train_{self.model_type.upper()}.py first."
            )
    
    def extract_features(self, img):
        """Extract features based on model type."""
        if self.model_type == 'svm':
            return extract_features_svm(img)
        else:
            return extract_features_knn(img)
    
    def predict(self, img):
        """
        Predict material class with confidence and rejection mechanism.
        
        Args:
            img: Input BGR image (will be resized to IMG_SIZE)
        
        Returns:
            tuple: (predicted_class_name, confidence, is_rejected)
        """
        # Resize image
        img = cv2.resize(img, IMG_SIZE)
        
        # Extract features
        features = self.extract_features(img)
        
        # Scale and PCA transform
        features_scaled = self.scaler.transform([features])
        features_pca = self.pca.transform(features_scaled)
        
        if self.model_type == 'svm':
            return self._predict_svm(features_pca)
        else:
            return self._predict_knn(features_pca)
    
    def _predict_svm(self, features_pca):
        """SVM prediction with probability-based rejection."""
        # Get prediction and probabilities
        prediction = self.model.predict(features_pca)[0]
        probabilities = self.model.predict_proba(features_pca)[0]
        
        max_prob = probabilities.max()
        confidence = max_prob
        
        # Get threshold
        threshold = self.threshold if isinstance(self.threshold, float) else SVM_CONFIDENCE_THRESHOLD
        
        # Check rejection
        if max_prob < threshold:
            return "unknown", confidence, True
        
        class_name = ID_TO_CLASS[prediction]
        return class_name, confidence, False
    
    def _predict_knn(self, features_pca):
        """k-NN prediction with distance and voting-based rejection."""
        # Get prediction
        prediction = self.model.predict(features_pca)[0]
        
        # Get distances to neighbors
        distances, indices = self.model.kneighbors(features_pca)
        avg_distance = distances.mean()
        
        # Calculate voting confidence
        neighbor_labels = self.model._y[indices[0]]
        voting_confidence = np.mean(neighbor_labels == prediction)
        
        # Get thresholds
        if isinstance(self.threshold, dict):
            dist_thresh = self.threshold.get('distance_threshold', 1.0)
            conf_thresh = self.threshold.get('confidence_threshold', KNN_CONFIDENCE_THRESHOLD)
        else:
            dist_thresh = 1.0
            conf_thresh = KNN_CONFIDENCE_THRESHOLD
        
        # Combined confidence (inverse of normalized distance * voting confidence)
        confidence = voting_confidence * (1 - min(avg_distance / (dist_thresh * 2), 1.0))
        
        # Check rejection
        if avg_distance > dist_thresh or voting_confidence < conf_thresh:
            return "unknown", confidence, True
        
        class_name = ID_TO_CLASS[prediction]
        return class_name, confidence, False


def draw_prediction(frame, class_name, confidence, is_rejected, fps=0):
    """
    Draw prediction results on frame with visual feedback.
    
    Args:
        frame: Input frame to draw on
        class_name: Predicted class name
        confidence: Confidence score (0-1)
        is_rejected: Whether prediction was rejected
        fps: Frames per second
    """
    h, w = frame.shape[:2]
    
    # Get color based on status
    if is_rejected:
        color = COLORS.get('unknown', (0, 255, 255))
        status_text = "UNCERTAIN"
    else:
        color = COLORS.get(class_name, (255, 255, 255))
        status_text = "CONFIDENT"
    
    # Draw background rectangle
    cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (350, 120), color, 2)
    
    # Draw class name
    cv2.putText(
        frame,
        f"Class: {class_name.upper()}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )
    
    # Draw confidence bar
    bar_width = int(300 * confidence)
    cv2.rectangle(frame, (20, 60), (320, 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 60), (20 + bar_width, 80), color, -1)
    cv2.putText(
        frame,
        f"Confidence: {confidence*100:.1f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )
    
    # Draw status
    cv2.putText(
        frame,
        status_text,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )
    
    # Draw FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (280, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )
    
    # Draw instructions at bottom
    cv2.putText(
        frame,
        "Press 'q' to quit | 'm' to switch model | 's' to screenshot",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1
    )
    
    return frame


def run_realtime_classification(model_type='svm', camera_id=0):
    """
    Run real-time material classification using webcam.
    
    Args:
        model_type: 'svm' or 'knn'
        camera_id: Camera device ID
    """
    print(f"\nStarting real-time classification with {model_type.upper()} model...")
    print("Press 'q' to quit, 'm' to switch model, 's' to save screenshot\n")
    
    # Initialize classifier
    try:
        classifier = MaterialClassifier(model_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Prediction smoothing (temporal averaging)
    prediction_history = []
    history_size = 5
    
    current_model = model_type
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Calculate FPS
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
        
        # Make prediction
        class_name, confidence, is_rejected = classifier.predict(frame)
        
        # Add to history for smoothing
        prediction_history.append((class_name, confidence, is_rejected))
        if len(prediction_history) > history_size:
            prediction_history.pop(0)
        
        # Get most common prediction (simple temporal smoothing)
        if len(prediction_history) >= 3:
            class_votes = {}
            for c, conf, rej in prediction_history:
                if c not in class_votes:
                    class_votes[c] = []
                class_votes[c].append((conf, rej))
            
            # Find most common class
            most_common = max(class_votes.keys(), key=lambda x: len(class_votes[x]))
            avg_conf = np.mean([c[0] for c in class_votes[most_common]])
            most_rejected = sum([c[1] for c in class_votes[most_common]]) > len(class_votes[most_common]) / 2
            
            class_name, confidence, is_rejected = most_common, avg_conf, most_rejected
        
        # Draw prediction on frame
        frame = draw_prediction(frame, class_name, confidence, is_rejected, fps)
        
        # Show model type
        cv2.putText(
            frame,
            f"Model: {current_model.upper()}",
            (frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )
        
        # Display frame
        cv2.imshow("Material Stream Identification System", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        
        elif key == ord('m'):
            # Switch model
            current_model = 'knn' if current_model == 'svm' else 'svm'
            print(f"Switching to {current_model.upper()} model...")
            try:
                classifier = MaterialClassifier(current_model)
                prediction_history = []
            except FileNotFoundError as e:
                print(f"Error: {e}")
                current_model = 'svm' if current_model == 'knn' else 'knn'
        
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Material Stream Identification System - Real-Time Classification"
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['svm', 'knn'],
        default='svm',
        help='Model type to use (default: svm)'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    run_realtime_classification(args.model, args.camera)


if __name__ == "__main__":
    main()
