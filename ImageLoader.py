import os
import cv2
import numpy as np

IMG_SIZE = (128, 128)

def load_images(folder, label):
    images, labels = [], []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(label)
    return images, labels


def augment_images(images, labels):
    aug_images, aug_labels = [], []

    for img, label in zip(images, labels):
        # 1️⃣ Original
        aug_images.append(img)
        aug_labels.append(label)

        # 2️⃣ Horizontal flip (good for most materials)
        flipped = cv2.flip(img, 1)
        aug_images.append(flipped)
        aug_labels.append(label)

        # 3️⃣ One randomized realistic transform
        aug = img.copy()

        # Random brightness & contrast
        alpha = np.random.uniform(0.9, 1.1)   # contrast
        beta  = np.random.randint(-20, 20)    # brightness
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

        # Small rotation (camera variation)
        angle = np.random.uniform(-10, 10)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        aug_images.append(aug)
        aug_labels.append(label)

    return aug_images, aug_labels

