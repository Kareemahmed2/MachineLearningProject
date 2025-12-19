import os
import cv2
import numpy as np
from PIL import Image

IMG_SIZE = (128, 128)


def load_images(folder, label):
    """
    Load images with robust error handling (PIL fallback for corrupted images)
    Returns: images (list), labels (list)
    """
    images, labels = [], []

    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist!")
        return images, labels

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    failed = 0

    for file in os.listdir(folder):
        if not file.lower().endswith(valid_extensions):
            continue

        path = os.path.join(folder, file)

        # Try OpenCV first
        img = cv2.imread(path)

        # If OpenCV fails, try PIL (handles corrupted images better)
        if img is None:
            try:
                pil_img = Image.open(path).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                failed += 1
                continue

        # Validate image
        if img is None or img.size == 0:
            failed += 1
            continue

        # Resize
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(label)

    if failed > 0:
        print(f"  ⚠️  Failed to load {failed} files")

    return images, labels


def balance_dataset(images, labels, target_count=500):
    """
    Balance dataset by augmenting to target_count per class
    Returns: np.array of images, np.array of labels
    """
    images = list(images)
    labels = list(labels)

    unique_classes = np.unique(labels)
    final_images = []
    final_labels = []

    print(f"Balancing to {target_count} samples per class...")

    for cls in unique_classes:
        # Get images for this class
        cls_indices = [i for i, l in enumerate(labels) if l == cls]
        cls_images = [images[i] for i in cls_indices]

        print(f"  Class {cls}: {len(cls_images)} original → ", end="")

        # Add all originals
        final_images.extend(cls_images)
        final_labels.extend([cls] * len(cls_images))

        # Calculate how many augmented needed
        needed = target_count - len(cls_images)

        if needed > 0:
            # Generate augmented images
            generated = 0
            while generated < needed:
                # Pick random image from this class
                idx = np.random.randint(0, len(cls_images))
                img = cls_images[idx]

                # Apply augmentation
                aug = augment_single_image(img)
                final_images.append(aug)
                final_labels.append(cls)
                generated += 1

        print(f"{target_count} total")

    # Convert to arrays
    final_images = np.array(final_images)
    final_labels = np.array(final_labels)

    # Shuffle
    indices = np.random.permutation(len(final_labels))
    final_images = final_images[indices]
    final_labels = final_labels[indices]

    return final_images, final_labels


def augment_single_image(img):
    """
    Apply realistic augmentation to a single image
    Returns: augmented image
    """
    aug = img.copy()

    # 1. Brightness & contrast (simulate different lighting)
    alpha = np.random.uniform(0.7, 1.3)  # contrast
    beta = np.random.randint(-40, 40)  # brightness
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # 2. Color shift in HSV (simulate different camera settings)
    if np.random.random() > 0.3:
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] *= np.random.uniform(0.9, 1.1)  # Hue
        hsv[..., 1] *= np.random.uniform(0.8, 1.2)  # Saturation
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. Rotation (camera angle variation)
    angle = np.random.uniform(-30, 30)
    h, w = aug.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 4. Flips
    if np.random.random() > 0.5:
        aug = cv2.flip(aug, 1)  # horizontal
    if np.random.random() > 0.5:
        aug = cv2.flip(aug, 0)  # vertical

    # 5. Blur (simulate different focus)
    if np.random.random() > 0.7:
        aug = cv2.GaussianBlur(aug, (5, 5), 0)

    # 6. Noise (simulate sensor noise)
    if np.random.random() > 0.7:
        noise = np.random.normal(0, 10, aug.shape)
        aug = np.clip(aug + noise, 0, 255).astype(np.uint8)

    # 7. Zoom/Scale
    if np.random.random() > 0.5:
        scale = np.random.uniform(0.9, 1.1)
        M_scale = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        aug = cv2.warpAffine(aug, M_scale, (w, h), borderMode=cv2.BORDER_REFLECT)

    return aug


# Optional: Function to augment immediately after loading (like your original approach)
def augment_images(images, labels):
    """
    Simple augmentation: each image → 3 versions (original + flip + random transform)
    This matches your original approach but with better augmentation
    """
    aug_images, aug_labels = [], []

    for img, label in zip(images, labels):
        # 1. Original
        aug_images.append(img)
        aug_labels.append(label)

        # 2. Horizontal flip
        flipped = cv2.flip(img, 1)
        aug_images.append(flipped)
        aug_labels.append(label)

        # 3. Random augmentation
        aug = augment_single_image(img)
        aug_images.append(aug)
        aug_labels.append(label)

    return aug_images, aug_labels