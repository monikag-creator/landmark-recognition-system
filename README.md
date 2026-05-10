# Landmark Recognition System using Deep Learning (CNN)

**Author:** Monika | AI/ML Program  
**Inspired by:** Google Street View Landmark Recognition  
**Architecture:** Custom CNN with 4 Convolutional Blocks + Global Average Pooling

---

## Project Overview

This project implements a **landmark recognition system** using Convolutional Neural Networks (CNNs). Inspired by the Google Street View landmark recognition challenge, the system identifies famous world landmarks from images.

**Landmarks Recognized:** Eiffel Tower | Colosseum | Taj Mahal | Big Ben | Statue of Liberty

---

## Dataset Details

| Property | Value |
|----------|-------|
| Classes | 5 landmark categories |
| Total images | 600 (480 train / 120 val) |
| Image size | 64×64 RGB |
| Train/Val split | 80% / 20% |
| Augmentation | Rotation, zoom, flip, brightness, shear |

---

## Model Architecture — Custom CNN

```
Input (64×64×3)
  → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
  → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
  → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.30)
  → Conv2D(256) → GlobalAveragePooling2D
  → Dense(256, relu) → Dropout(0.40)
  → Dense(128, relu)
  → Dense(5, softmax)
```
Total Parameters: ~488,645 (1.86 MB)

---

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **100%** |
| Weighted Precision | **1.00** |
| Weighted Recall | **1.00** |
| Weighted F1-Score | **1.00** |

---

## Steps to Run

```bash
# Install dependencies
pip install tensorflow scikit-learn matplotlib seaborn pillow numpy

# Train (demo mode generates synthetic data automatically)
python train.py --demo

# Predict on a new image
python predict.py --image path/to/image.jpg
```

---

## Potential Improvements
- Use real Google Landmarks Dataset v2 (5M+ images)
- Apply MobileNetV2 / EfficientNet transfer learning with ImageNet weights
- Add Grad-CAM activation maps for interpretability
- Deploy as REST API (FastAPI + Docker)
