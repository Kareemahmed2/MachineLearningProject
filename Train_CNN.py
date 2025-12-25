import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from ImageLoader import load_images, balance_dataset
from CNN_Model import build_cnn

# -----------------------------
# CONFIG
# -----------------------------
EPOCHS = 25
BATCH_SIZE = 32
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

# -----------------------------
# LOAD DATA
# -----------------------------
def load_all_data(base_path="./test_set"):
    images = []
    labels = []

    for label, name in enumerate(CLASS_NAMES):
        x, y = load_images(f"{base_path}/{name}", label)
        images.extend(x)
        labels.extend(y)

    return np.array(images), np.array(labels)


print("Loading dataset...")
images, labels = load_all_data()

# Normalize images
images = images.astype("float32") / 255.0

# Train / Test split
x_train, x_test, y_train, y_test = train_test_split(
    images, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# Balance training set
print("Balancing dataset...")
x_train, y_train = balance_dataset(x_train, y_train, target_count=500)

# -----------------------------
# DATA AUGMENTATION (ON-THE-FLY)
# -----------------------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)

datagen.fit(x_train)

# -----------------------------
# TRAIN CNN
# -----------------------------
model = build_cnn()
model.summary()

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(x_test, y_test)
)

# -----------------------------
# EVALUATION
# -----------------------------
pred = np.argmax(model.predict(x_test), axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=CLASS_NAMES))

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("cnn_model.h5")
print("\n✓ CNN model saved as cnn_model.h5")
