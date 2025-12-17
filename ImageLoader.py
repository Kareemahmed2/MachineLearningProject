"""
Image Loading and Augmentation - Balanced Version
Not too aggressive to preserve class characteristics
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import random

from config import (
    IMG_SIZE, DATASET_PATH, CLASS_NAMES, PRIMARY_CLASSES,
    TARGET_SAMPLES_PER_CLASS, MIN_AUGMENTATION_FACTOR, CLASS_TO_ID
)


def load_images_from_folder(folder: str, label: int, max_images: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
    """Load images from folder."""
    images, labels = [], []
    
    if not os.path.exists(folder):
        return images, labels
    
    files = os.listdir(folder)
    if max_images:
        files = files[:max_images]
    
    for filename in files:
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(label)
    
    return images, labels


def load_dataset(include_unknown: bool = True, balance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load complete dataset."""
    all_images, all_labels = [], []
    classes_to_load = CLASS_NAMES if include_unknown else PRIMARY_CLASSES
    
    print("Loading dataset...")
    for class_name in classes_to_load:
        folder = os.path.join(DATASET_PATH, class_name)
        label = CLASS_TO_ID[class_name]
        images, labels = load_images_from_folder(folder, label)
        all_images.extend(images)
        all_labels.extend(labels)
        print(f"  {class_name}: {len(images)} images")
    
    print(f"Total: {len(all_images)} images")
    return np.array(all_images), np.array(all_labels)


def apply_augmentation(img: np.ndarray, aug_type: str = 'random') -> np.ndarray:
    """Apply single augmentation."""
    aug = img.copy()
    
    if aug_type == 'random':
        aug_type = random.choice(['flip', 'rotate', 'brightness', 'blur', 'combined'])
    
    if aug_type == 'flip':
        aug = cv2.flip(aug, random.choice([0, 1]))
    
    elif aug_type == 'rotate':
        angle = random.uniform(-15, 15)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 'brightness':
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-25, 25)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    
    elif aug_type == 'blur':
        if random.random() > 0.5:
            aug = cv2.GaussianBlur(aug, (3, 3), 0)
    
    elif aug_type == 'combined':
        # Flip + small rotation + brightness
        if random.random() > 0.5:
            aug = cv2.flip(aug, 1)
        
        angle = random.uniform(-10, 10)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        alpha = random.uniform(0.9, 1.1)
        beta = random.randint(-15, 15)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    
    return aug


def augment_images(images: List[np.ndarray], labels: List[int], 
                   target_per_class: int = TARGET_SAMPLES_PER_CLASS) -> Tuple[List[np.ndarray], List[int]]:
    """Balance and augment dataset."""
    images, labels = list(images), list(labels)
    
    unique_labels = set(labels)
    class_images = {label: [] for label in unique_labels}
    for img, label in zip(images, labels):
        class_images[label].append(img)
    
    aug_images, aug_labels = [], []
    
    print("\nAugmenting dataset...")
    for label in sorted(unique_labels):
        class_imgs = class_images[label]
        current_count = len(class_imgs)
        
        # Add originals
        aug_images.extend(class_imgs)
        aug_labels.extend([label] * current_count)
        
        # Calculate needed augmentations
        needed = max(target_per_class - current_count, int(current_count * 0.3))
        
        # Generate augmented samples
        for _ in range(needed):
            img = random.choice(class_imgs)
            aug_img = apply_augmentation(img)
            aug_images.append(aug_img)
            aug_labels.append(label)
        
        final = len([l for l in aug_labels if l == label])
        print(f"  Class {label}: {current_count} -> {final}")
    
    print(f"Total after augmentation: {len(aug_images)}")
    return aug_images, aug_labels


def generate_unknown_samples(images: List[np.ndarray], num_samples: int = 200) -> List[np.ndarray]:
    """Generate unknown class samples."""
    unknown = []
    for _ in range(num_samples):
        img = random.choice(images).copy()
        
        # Heavy distortion
        distort = random.choice(['blur', 'noise', 'crop', 'color'])
        
        if distort == 'blur':
            img = cv2.GaussianBlur(img, (21, 21), 0)
        elif distort == 'noise':
            noise = np.random.normal(0, 40, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif distort == 'crop':
            h, w = img.shape[:2]
            x1, y1 = random.randint(0, w//3), random.randint(0, h//3)
            x2, y2 = random.randint(2*w//3, w), random.randint(2*h//3, h)
            img = cv2.resize(img[y1:y2, x1:x2], IMG_SIZE)
        else:
            img = img.astype(np.float32)
            for i in range(3):
                img[:,:,i] *= random.uniform(0.4, 1.8)
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        unknown.append(img)
    
    return unknown


# Backward compatibility
def load_images(folder: str, label: int) -> Tuple[List[np.ndarray], List[int]]:
    return load_images_from_folder(folder, label)
