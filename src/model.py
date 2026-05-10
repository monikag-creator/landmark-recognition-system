"""
Landmark Recognition Model using Transfer Learning (MobileNetV2)
Author: Monika | AI/ML Program
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001


def build_model(num_classes: int) -> Model:
    """
    Build landmark recognition model using MobileNetV2 as backbone.
    
    Args:
        num_classes: Number of landmark categories to classify
    Returns:
        Compiled Keras model
    """
    # Load MobileNetV2 pre-trained on ImageNet, without top classification layer
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers initially (feature extraction phase)
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"[INFO] Model built — Total params: {model.count_params():,}")
    print(f"[INFO] Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    return model


def unfreeze_and_fine_tune(model: Model, fine_tune_at: int = 100) -> Model:
    """
    Unfreeze top layers of base model for fine-tuning.
    
    Args:
        model: Previously compiled model
        fine_tune_at: Layer index from which to unfreeze
    Returns:
        Re-compiled model with fine-tuning enabled
    """
    base_model = model.layers[1]  # MobileNetV2 is the second layer
    base_model.trainable = True

    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),  # Lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"[INFO] Fine-tuning enabled from layer {fine_tune_at}")
    print(f"[INFO] Trainable params after unfreeze: {trainable_count:,}")
    return model


def get_callbacks(checkpoint_path: str) -> list:
    """Return list of training callbacks."""
    return [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
