"""
Image Loading and Data Augmentation Module
Handles loading images from dataset and applying augmentation techniques
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import random

from config import (
    IMG_SIZE, DATASET_PATH, CLASS_NAMES, PRIMARY_CLASSES,
    TARGET_SAMPLES_PER_CLASS, MIN_AUGMENTATION_FACTOR,
    AUG_ROTATION_RANGE, AUG_BRIGHTNESS_RANGE, AUG_CONTRAST_RANGE,
    AUG_BLUR_PROBABILITY, AUG_NOISE_PROBABILITY, CLASS_TO_ID
)


def load_images_from_folder(folder: str, label: int, max_images: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load images from a folder and assign a label.
    
    Args:
        folder: Path to the folder containing images
        label: Class label to assign to all images
        max_images: Maximum number of images to load (None for all)
    
    Returns:
        Tuple of (images list, labels list)
    """
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
            
        # Resize to standard size
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(label)
    
    return images, labels


def load_dataset(include_unknown: bool = True, balance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the complete dataset with optional class balancing.
    
    Args:
        include_unknown: Whether to include the unknown class
        balance: Whether to balance class sizes
    
    Returns:
        Tuple of (images array, labels array)
    """
    all_images = []
    all_labels = []
    class_counts = {}
    
    # Determine which classes to load
    classes_to_load = CLASS_NAMES if include_unknown else PRIMARY_CLASSES
    
    print("Loading dataset...")
    for class_name in classes_to_load:
        folder = os.path.join(DATASET_PATH, class_name)
        label = CLASS_TO_ID[class_name]
        
        images, labels = load_images_from_folder(folder, label)
        class_counts[class_name] = len(images)
        
        all_images.extend(images)
        all_labels.extend(labels)
        
        print(f"  {class_name}: {len(images)} images")
    
    print(f"Total: {len(all_images)} images")
    
    return np.array(all_images), np.array(all_labels)


def apply_augmentation(img: np.ndarray) -> np.ndarray:
    """
    Apply a single random augmentation to an image.
    
    Args:
        img: Input image (BGR format)
    
    Returns:
        Augmented image
    """
    aug = img.copy()
    
    # Random choice of augmentation
    aug_type = random.choice(['flip', 'rotate', 'brightness', 'blur', 'noise', 'combined'])
    
    if aug_type == 'flip':
        # Horizontal or vertical flip
        flip_code = random.choice([0, 1, -1])  # 0=vertical, 1=horizontal, -1=both
        aug = cv2.flip(aug, flip_code)
    
    elif aug_type == 'rotate':
        # Random rotation
        angle = random.uniform(*AUG_ROTATION_RANGE)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 'brightness':
        # Random brightness and contrast
        alpha = random.uniform(*AUG_CONTRAST_RANGE)
        beta = random.randint(-30, 30)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    
    elif aug_type == 'blur':
        # Gaussian blur
        ksize = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)
    
    elif aug_type == 'noise':
        # Gaussian noise
        noise = np.random.normal(0, 10, aug.shape).astype(np.int16)
        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    elif aug_type == 'combined':
        # Apply multiple augmentations
        # Flip
        if random.random() > 0.5:
            aug = cv2.flip(aug, 1)
        
        # Rotate
        angle = random.uniform(-10, 10)
        h, w = aug.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Brightness
        alpha = random.uniform(0.9, 1.1)
        beta = random.randint(-20, 20)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)
    
    return aug


def augment_images(images: List[np.ndarray], labels: List[int], 
                   target_per_class: int = TARGET_SAMPLES_PER_CLASS) -> Tuple[List[np.ndarray], List[int]]:
    """
    Augment images to balance classes and meet minimum augmentation requirement.
    
    Args:
        images: List of input images
        labels: List of corresponding labels
        target_per_class: Target number of samples per class
    
    Returns:
        Tuple of (augmented images, augmented labels)
    """
    images = list(images)
    labels = list(labels)
    
    # Count images per class
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
        
        # Calculate how many augmented images we need
        min_needed = int(current_count * (MIN_AUGMENTATION_FACTOR - 1))  # At least 30% more
        target_needed = max(target_per_class - current_count, min_needed)
        
        if target_needed > 0:
            # Generate augmented images
            for _ in range(target_needed):
                # Pick random image from class
                img = random.choice(class_imgs)
                aug_img = apply_augmentation(img)
                aug_images.append(aug_img)
                aug_labels.append(label)
        
        print(f"  Class {label}: {current_count} -> {current_count + max(0, target_needed)}")
    
    print(f"Total after augmentation: {len(aug_images)}")
    
    return aug_images, aug_labels


def generate_unknown_samples(images: List[np.ndarray], num_samples: int = 200) -> List[np.ndarray]:
    """
    Generate 'unknown' class samples by heavily distorting existing images.
    
    Args:
        images: Source images to distort
        num_samples: Number of unknown samples to generate
    
    Returns:
        List of distorted images for unknown class
    """
    unknown_images = []
    
    for _ in range(num_samples):
        # Pick random image
        img = random.choice(images).copy()
        
        # Apply heavy distortion - choose one or combine
        distortion = random.choice(['heavy_blur', 'noise', 'partial', 'color_shift', 'combined'])
        
        if distortion == 'heavy_blur':
            # Very heavy blur
            ksize = random.choice([15, 21, 31])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        elif distortion == 'noise':
            # Heavy noise
            noise = np.random.normal(0, 50, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        elif distortion == 'partial':
            # Show only partial image (crop and resize)
            h, w = img.shape[:2]
            x1 = random.randint(0, w//2)
            y1 = random.randint(0, h//2)
            x2 = random.randint(w//2, w)
            y2 = random.randint(h//2, h)
            img = cv2.resize(img[y1:y2, x1:x2], IMG_SIZE)
        
        elif distortion == 'color_shift':
            # Extreme color shift
            img = img.astype(np.float32)
            img[:,:,0] = np.clip(img[:,:,0] * random.uniform(0.3, 2.0), 0, 255)
            img[:,:,1] = np.clip(img[:,:,1] * random.uniform(0.3, 2.0), 0, 255)
            img[:,:,2] = np.clip(img[:,:,2] * random.uniform(0.3, 2.0), 0, 255)
            img = img.astype(np.uint8)
        
        elif distortion == 'combined':
            # Multiple distortions
            # Blur
            img = cv2.GaussianBlur(img, (11, 11), 0)
            # Noise
            noise = np.random.normal(0, 25, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Color shift
            img = img.astype(np.float32)
            img[:,:,random.randint(0,2)] *= random.uniform(0.5, 1.5)
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        unknown_images.append(img)
    
    return unknown_images


def create_unknown_class_folder(source_images: List[np.ndarray], 
                                 num_samples: int = 200) -> None:
    """
    Create unknown class folder with generated samples.
    
    Args:
        source_images: Source images to distort
        num_samples: Number of samples to generate
    """
    unknown_folder = os.path.join(DATASET_PATH, "unknown")
    os.makedirs(unknown_folder, exist_ok=True)
    
    unknown_images = generate_unknown_samples(source_images, num_samples)
    
    for i, img in enumerate(unknown_images):
        filepath = os.path.join(unknown_folder, f"unknown_{i:04d}.jpg")
        cv2.imwrite(filepath, img)
    
    print(f"Created {len(unknown_images)} unknown samples in {unknown_folder}")


# For backwards compatibility
def load_images(folder: str, label: int) -> Tuple[List[np.ndarray], List[int]]:
    """Legacy function for loading images."""
    return load_images_from_folder(folder, label)
