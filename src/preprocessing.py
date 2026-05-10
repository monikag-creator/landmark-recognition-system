"""
Data Preprocessing & Augmentation Pipeline for Landmark Recognition
Author: Monika | AI/ML Program
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_data_generators(data_dir: str, val_split: float = 0.2):
    """
    Create training and validation data generators with augmentation.
    
    Dataset structure expected:
        data_dir/
            class_1/image1.jpg ...
            class_2/image1.jpg ...
            ...
    
    Args:
        data_dir: Root directory containing class subfolders
        val_split: Fraction of data for validation
    Returns:
        Tuple of (train_gen, val_gen, class_names)
    """
    # Training generator with heavy augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=val_split
    )

    # Validation generator — NO augmentation, only normalization
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=val_split
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    class_names = list(train_gen.class_indices.keys())
    print(f"[INFO] Classes found: {len(class_names)} → {class_names}")
    print(f"[INFO] Training samples: {train_gen.samples}")
    print(f"[INFO] Validation samples: {val_gen.samples}")
    return train_gen, val_gen, class_names


def preprocess_single_image(img_path: str) -> np.ndarray:
    """
    Preprocess a single image for inference.
    
    Args:
        img_path: Path to the image file
    Returns:
        Preprocessed numpy array of shape (1, 224, 224, 3)
    """
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def visualize_augmentation(data_dir: str, save_path: str = None):
    """Show original vs augmented versions of a sample image."""
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    # Pick first available image
    first_class = os.listdir(data_dir)[0]
    first_img_name = os.listdir(os.path.join(data_dir, first_class))[0]
    img_path = os.path.join(data_dir, first_class, first_img_name)

    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle("Data Augmentation Samples", fontsize=16, fontweight='bold', y=1.01)

    axes[0][0].imshow(img)
    axes[0][0].set_title("Original", fontweight='bold')
    axes[0][0].axis('off')

    for idx, aug_img in enumerate(datagen.flow(img_array, batch_size=1)):
        row, col = divmod(idx + 1, 5)
        if row >= 2:
            break
        axes[row][col].imshow(aug_img[0].astype('uint8'))
        axes[row][col].set_title(f"Augmented #{idx+1}")
        axes[row][col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Augmentation visualization saved → {save_path}")


def create_synthetic_dataset(save_dir: str, num_classes: int = 5, images_per_class: int = 60):
    """
    Generate a synthetic dataset of colored landmark silhouettes for demo/testing.
    Each class = distinct color + shape pattern representing a landmark type.
    
    Args:
        save_dir: Directory to save generated images
        num_classes: Number of landmark classes
        images_per_class: Number of images per class
    """
    import random
    from PIL import Image, ImageDraw, ImageFont

    landmark_classes = [
        "Eiffel_Tower", "Colosseum", "Taj_Mahal", "Big_Ben", "Statue_of_Liberty",
        "Sagrada_Familia", "Burj_Khalifa", "Sydney_Opera", "Machu_Picchu", "Great_Wall"
    ][:num_classes]

    # Unique base color per class
    class_colors = [
        (70, 130, 180), (220, 80, 80), (80, 160, 80),
        (200, 160, 50), (150, 80, 200), (60, 180, 180),
        (240, 120, 40), (100, 100, 200), (180, 100, 60), (60, 140, 80)
    ]

    os.makedirs(save_dir, exist_ok=True)
    random.seed(42)

    for cls_idx, cls_name in enumerate(landmark_classes):
        cls_dir = os.path.join(save_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        base_color = class_colors[cls_idx % len(class_colors)]

        for i in range(images_per_class):
            # Slight color variation per image
            r = np.clip(base_color[0] + random.randint(-20, 20), 0, 255)
            g = np.clip(base_color[1] + random.randint(-20, 20), 0, 255)
            b = np.clip(base_color[2] + random.randint(-20, 20), 0, 255)

            img = Image.new("RGB", IMG_SIZE, (r, g, b))
            draw = ImageDraw.Draw(img)

            # Draw unique pattern per class (simulates landmark silhouette)
            cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2
            pattern_color = (255, 255, 255)

            if cls_idx % 5 == 0:  # Tower-like
                draw.rectangle([cx-15, cy-70, cx+15, cy+50], fill=pattern_color)
                draw.polygon([(cx-30, cy-70), (cx, cy-110), (cx+30, cy-70)], fill=pattern_color)
            elif cls_idx % 5 == 1:  # Arch/dome
                draw.ellipse([cx-60, cy-60, cx+60, cy+60], fill=pattern_color)
                draw.rectangle([cx-60, cy, cx+60, cy+60], fill=(r, g, b))
            elif cls_idx % 5 == 2:  # Pyramid
                draw.polygon([(cx, cy-80), (cx-80, cy+50), (cx+80, cy+50)], fill=pattern_color)
            elif cls_idx % 5 == 3:  # Palace / wide
                draw.rectangle([cx-80, cy-40, cx+80, cy+50], fill=pattern_color)
                draw.rectangle([cx-20, cy-70, cx+20, cy-40], fill=pattern_color)
            else:  # Bridge/gate
                draw.arc([cx-70, cy-50, cx+70, cy+50], start=180, end=0, fill=pattern_color, width=8)
                draw.line([(cx-70, cy-50), (cx-70, cy+50)], fill=pattern_color, width=8)
                draw.line([(cx+70, cy-50), (cx+70, cy+50)], fill=pattern_color, width=8)

            # Add noise for realism
            img_array = np.array(img)
            noise = np.random.normal(0, 12, img_array.shape).astype(np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(img_array).save(os.path.join(cls_dir, f"img_{i:04d}.jpg"))

    print(f"[INFO] Synthetic dataset created: {num_classes} classes × {images_per_class} images = {num_classes * images_per_class} total")
    return landmark_classes
