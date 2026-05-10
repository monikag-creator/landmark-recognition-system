"""
Main Training Script — Landmark Recognition System
Author: Monika | AI/ML Program

Usage:
    python train.py --data_dir data/landmarks --epochs 20
    python train.py --demo   (runs with synthetic dataset)
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.preprocessing import get_data_generators, create_synthetic_dataset, visualize_augmentation
from src.model import build_model, unfreeze_and_fine_tune, get_callbacks
from src.evaluate import evaluate_model, plot_training_history, plot_confusion_matrix, \
                         plot_per_class_metrics, visualize_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Landmark Recognition Training")
    parser.add_argument('--data_dir', type=str, default='data/landmarks',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--demo', action='store_true',
                        help='Run with synthetic demo dataset')
    parser.add_argument('--fine_tune', action='store_true', default=True,
                        help='Enable fine-tuning phase')
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  LANDMARK RECOGNITION SYSTEM — Transfer Learning")
    print("  Using MobileNetV2 + Custom Classification Head")
    print("="*60 + "\n")

    # Directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Step 1: Dataset
    data_dir = args.data_dir
    if args.demo or not os.path.exists(data_dir):
        print("[SETUP] Generating synthetic demo dataset...")
        data_dir = 'data/synthetic_landmarks'
        class_names = create_synthetic_dataset(
            save_dir=data_dir,
            num_classes=5,
            images_per_class=80
        )
    else:
        class_names = None

    # Step 2: Data generators
    train_gen, val_gen, class_names = get_data_generators(data_dir, val_split=0.2)
    num_classes = len(class_names)

    # Visualize augmentation
    visualize_augmentation(data_dir, save_path='results/augmentation_samples.png')

    # Step 3: Build model
    print("\n[MODEL] Building MobileNetV2-based model...")
    model = build_model(num_classes)
    model.summary()

    # Step 4: Phase 1 — Feature extraction (frozen base)
    print("\n[TRAIN] Phase 1: Feature Extraction (frozen base)...")
    phase1_epochs = min(args.epochs // 2, 8)
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase1_epochs,
        callbacks=get_callbacks('models/best_model_phase1.keras'),
        verbose=1
    )

    # Step 5: Phase 2 — Fine-tuning (unfreeze top layers)
    if args.fine_tune:
        print("\n[TRAIN] Phase 2: Fine-Tuning (unfreezing top layers)...")
        model = unfreeze_and_fine_tune(model, fine_tune_at=100)
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs - phase1_epochs,
            callbacks=get_callbacks('models/best_model_final.keras'),
            verbose=1
        )
        # Merge histories
        combined_history_data = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        }
        class FakeHistory:
            def __init__(self, data): self.history = data
        history = FakeHistory(combined_history_data)
    else:
        history = history1

    # Step 6: Plots
    plot_training_history(history, save_path='results/training_history.png')

    # Step 7: Evaluate
    metrics = evaluate_model(model, val_gen, class_names)

    plot_confusion_matrix(metrics['y_true'], metrics['y_pred'], class_names,
                          save_path='results/confusion_matrix.png')
    plot_per_class_metrics(metrics['y_true'], metrics['y_pred'], class_names,
                           save_path='results/per_class_metrics.png')
    visualize_predictions(model, val_gen, class_names, num_images=12,
                          save_path='results/sample_predictions.png')

    # Step 8: Save model & metadata
    model.save('models/landmark_recognition_model.keras')
    metadata = {
        'class_names': class_names,
        'num_classes': num_classes,
        'input_size': [224, 224],
        'architecture': 'MobileNetV2 + Custom Head',
        'final_val_accuracy': float(metrics['accuracy']),
        'weighted_precision': float(metrics['precision']),
        'weighted_recall': float(metrics['recall']),
        'weighted_f1': float(metrics['f1'])
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print(f"  TRAINING COMPLETE")
    print(f"  Val Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  F1-Score     : {metrics['f1']:.4f}")
    print(f"  Model saved  : models/landmark_recognition_model.keras")
    print(f"  Results      : results/")
    print("="*60)


if __name__ == '__main__':
    main()
