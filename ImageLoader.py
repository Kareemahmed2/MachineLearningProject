"""
Image Loading and Data Augmentation Module - Enhanced Version
Stronger augmentation for better model generalization
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
    """Load images from a folder and assign a label."""
    images = []
    labels = []
    
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist")
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
    """Load the complete dataset."""
    all_images = []
    all_labels = []
    
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


def apply_strong_augmentation(img: np.ndarray) -> np.ndarray:
    """
    Apply strong augmentation for better generalization.
    """
    aug = img.copy()
    
    # Random horizontal flip (50%)
    if random.random() > 0.5:
        aug = cv2.flip(aug, 1)
    
    # Random rotation (-20 to +20 degrees)
    angle = random.uniform(-20, 20)
    h, w = aug.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness and contrast
    alpha = random.uniform(0.7, 1.3)  # Contrast
    beta = random.randint(-40, 40)    # Brightness
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    
    # Random blur (30% chance)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)
    
    # Random noise (20% chance)
    if random.random() < 0.2:
        noise = np.random.normal(0, 15, aug.shape).astype(np.int16)
        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Random color jitter (30% chance)
    if random.random() < 0.3:
        aug = aug.astype(np.float32)
        for i in range(3):
            aug[:,:,i] *= random.uniform(0.8, 1.2)
        aug = np.clip(aug, 0, 255).astype(np.uint8)
    
    # Random scale (20% chance)
    if random.random() < 0.2:
        scale = random.uniform(0.85, 1.15)
        new_h, new_w = int(h * scale), int(w * scale)
        aug = cv2.resize(aug, (new_w, new_h))
        
        # Pad or crop to original size
        if scale < 1:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            aug = cv2.copyMakeBorder(aug, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, 
                                      cv2.BORDER_REFLECT)
        else:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            aug = aug[start_h:start_h+h, start_w:start_w+w]
        
        aug = cv2.resize(aug, IMG_SIZE)
    
    return aug


def augment_images(images: List[np.ndarray], labels: List[int], 
                   target_per_class: int = TARGET_SAMPLES_PER_CLASS) -> Tuple[List[np.ndarray], List[int]]:
    """
    Augment images with strong augmentation for better generalization.
    """
    images = list(images)
    labels = list(labels)
    
    unique_labels = set(labels)
    class_images = {label: [] for label in unique_labels}
    
    for img, label in zip(images, labels):
        class_images[label].append(img)
    
    aug_images = []
    aug_labels = []
    
    print("\nAugmenting dataset...")
    for label in unique_labels:
        class_imgs = class_images[label]
        current_count = len(class_imgs)
        
        # Add original images
        aug_images.extend(class_imgs)
        aug_labels.extend([label] * current_count)
        
        # Calculate augmentation needed (at least 30% more, target ~500)
        min_needed = int(current_count * (MIN_AUGMENTATION_FACTOR - 1))
        target_needed = max(target_per_class - current_count, min_needed)
        
        # Generate multiple augmentations per original image
        if target_needed > 0:
            augmentations_per_image = max(1, target_needed // current_count)
            
            for img in class_imgs:
                for _ in range(augmentations_per_image):
                    if len([l for l in aug_labels if l == label]) >= target_per_class:
                        break
                    aug_img = apply_strong_augmentation(img)
                    aug_images.append(aug_img)
                    aug_labels.append(label)
            
            # Fill remaining if needed
            while len([l for l in aug_labels if l == label]) < target_per_class:
                img = random.choice(class_imgs)
                aug_img = apply_strong_augmentation(img)
                aug_images.append(aug_img)
                aug_labels.append(label)
        
        final_count = len([l for l in aug_labels if l == label])
        print(f"  Class {label}: {current_count} -> {final_count}")
    
    print(f"Total after augmentation: {len(aug_images)}")
    
    return aug_images, aug_labels


def generate_unknown_samples(images: List[np.ndarray], num_samples: int = 200) -> List[np.ndarray]:
    """Generate unknown class samples by distorting images."""
    unknown_images = []
    
    for _ in range(num_samples):
        img = random.choice(images).copy()
        
        distortion = random.choice(['blur', 'noise', 'partial', 'color', 'combined'])
        
        if distortion == 'blur':
            ksize = random.choice([15, 21, 31])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        elif distortion == 'noise':
            noise = np.random.normal(0, 50, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        elif distortion == 'partial':
            h, w = img.shape[:2]
            x1, y1 = random.randint(0, w//3), random.randint(0, h//3)
            x2, y2 = random.randint(2*w//3, w), random.randint(2*h//3, h)
            img = cv2.resize(img[y1:y2, x1:x2], IMG_SIZE)
        
        elif distortion == 'color':
            img = img.astype(np.float32)
            for i in range(3):
                img[:,:,i] *= random.uniform(0.3, 2.0)
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        else:  # combined
            img = cv2.GaussianBlur(img, (11, 11), 0)
            noise = np.random.normal(0, 25, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        unknown_images.append(img)
    
    return unknown_images


# Legacy compatibility
def load_images(folder: str, label: int) -> Tuple[List[np.ndarray], List[int]]:
    return load_images_from_folder(folder, label)
