"""
Model Evaluation, Metrics, and Visualization
Author: Monika | AI/ML Program
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
import tensorflow as tf


def plot_training_history(history, save_path: str = None):
    """Plot accuracy and loss curves from training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=16, fontweight='bold')

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='#2196F3')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#4CAF50', linestyle='--')
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2, color='#F44336')
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#FF9800', linestyle='--')
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training history plot saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str = None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Confusion Matrix — Landmark Recognition", fontsize=16, fontweight='bold')

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    plt.setp(axes[0].get_xticklabels(), rotation=40, ha='right', fontsize=9)

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    axes[1].set_title("Normalized (Recall per Class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.setp(axes[1].get_xticklabels(), rotation=40, ha='right', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


def plot_per_class_metrics(y_true, y_pred, class_names, save_path: str = None):
    """Bar chart of per-class precision, recall, F1."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#2196F3', alpha=0.85)
    ax.bar(x, recall, width, label='Recall', color='#4CAF50', alpha=0.85)
    ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800', alpha=0.85)

    ax.set_xlabel("Landmark Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics: Precision / Recall / F1-Score", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=10)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        ax.bar_label(bars, fmt='%.2f', fontsize=7.5, padding=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Per-class metrics saved → {save_path}")


def evaluate_model(model, val_gen, class_names):
    """
    Full model evaluation: loss, accuracy, classification report.
    
    Returns:
        dict: All computed metrics
    """
    print("\n[EVAL] Running model evaluation...")

    # Predictions
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes[:len(y_pred)]

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    print(f"\n{'='*55}")
    print(f"  Overall Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Weighted Precision: {precision:.4f}")
    print(f"  Weighted Recall   : {recall:.4f}")
    print(f"  Weighted F1-Score : {f1:.4f}")
    print(f"{'='*55}")
    print("\nPer-Class Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs
    }


def visualize_predictions(model, val_gen, class_names, num_images: int = 12, save_path: str = None):
    """
    Grid of sample predictions with confidence scores.
    Green border = correct, Red = wrong.
    """
    val_gen.reset()
    images, labels = next(val_gen)
    images = images[:num_images]
    labels = labels[:num_images]

    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)

    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=14, fontweight='bold')

    axes_flat = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for idx in range(num_images):
        ax = axes_flat[idx]
        # De-preprocess for display
        img_display = images[idx].copy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

        ax.imshow(img_display)
        correct = pred_classes[idx] == true_classes[idx]
        border_color = '#4CAF50' if correct else '#F44336'
        confidence = preds[idx][pred_classes[idx]] * 100

        ax.set_title(
            f"Pred: {class_names[pred_classes[idx]]}\nTrue: {class_names[true_classes[idx]]}\nConf: {confidence:.1f}%",
            fontsize=9,
            color='green' if correct else 'red',
            fontweight='bold'
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        ax.axis('off')

    # Hide unused axes
    for idx in range(num_images, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Prediction grid saved → {save_path}")
