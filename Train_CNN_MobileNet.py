import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ✅ IMPORT YOUR IMAGE LOADER
from ImageLoader import load_images, balance_dataset, IMG_SIZE

# ======================
# CONFIG
# ======================
DATASET_DIR = "test_set"
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 32
EPOCHS = 20

# ======================
# LOAD DATASET USING ImageLoader
# ======================
print("Loading dataset...")

images = []
labels = []

for label, class_name in enumerate(CLASS_NAMES):
    folder = os.path.join(DATASET_DIR, class_name)
    imgs, lbls = load_images(folder, label)
    images.extend(imgs)
    labels.extend(lbls)

images = np.array(images)
labels = np.array(labels)

# ======================
# BALANCE DATASET (YOUR FUNCTION)
# ======================
print("Balancing dataset...")
images, labels = balance_dataset(images, labels, target_count=500)

# ======================
# PREPROCESS FOR MOBILENET
# ======================
# BGR → RGB
images = images[..., ::-1]

# MobileNet preprocessing
images = preprocess_input(images.astype(np.float32))

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    images,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# ======================
# BUILD MOBILENETV2 MODEL
# ======================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# TRAIN
# ======================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ======================
# EVALUATE
# ======================
y_pred = np.argmax(model.predict(X_val), axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=CLASS_NAMES))

# ======================
# SAVE MODEL
# ======================
model.save("cnn_mobilenet.keras")
print("\n✓ MobileNetV2 model saved as cnn_mobilenet.keras")
