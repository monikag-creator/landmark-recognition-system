"""
predict.py
===========
Run landmark recognition on a single image or a batch folder.

Usage:
    # Single image
    python predict.py --image path/to/image.jpg --model models/landmark_model.h5

    # Batch folder
    python predict.py --folder path/to/images/ --model models/landmark_model.h5
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


IMG_SIZE = (224, 224)


def load_class_names(train_dir: str = "data/train") -> list:
    """Reads class names from the training folder structure."""
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    return sorted([d.name for d in train_path.iterdir() if d.is_dir()])


def preprocess_image(image_path: str) -> np.ndarray:
    img = keras_image.load_img(image_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_single(model, image_path: str, class_names: list, top_k: int = 3) -> dict:
    """Returns prediction dict with landmark name, confidence, and top-k alternatives."""
    arr   = preprocess_image(image_path)
    preds = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:top_k]

    result = {
        "image":      str(image_path),
        "prediction": class_names[top_idx[0]],
        "confidence": f"{preds[top_idx[0]] * 100:.2f}%",
        "top_k": [
            {
                "rank":       i + 1,
                "landmark":   class_names[idx],
                "confidence": f"{preds[idx] * 100:.2f}%"
            }
            for i, idx in enumerate(top_idx)
        ]
    }
    return result


def predict_batch(model, folder_path: str, class_names: list) -> list:
    """Runs inference on all images in a folder."""
    folder = Path(folder_path)
    images = sorted([
        f for f in folder.glob("**/*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    results = []
    for img_path in images:
        result = predict_single(model, str(img_path), class_names)
        results.append(result)
        print(f"  {img_path.name:<40} → {result['prediction']:30s}  ({result['confidence']})")

    return results


def save_results(results: list, save_path: str = "results/predictions.json") -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅  Predictions saved → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmark recognition inference")
    parser.add_argument("--model",     default="models/landmark_model.h5",
                        help="Path to trained .h5 model")
    parser.add_argument("--train_dir", default="data/train",
                        help="Training directory to infer class names from")
    parser.add_argument("--image",     default=None,
                        help="Single image path for inference")
    parser.add_argument("--folder",    default=None,
                        help="Folder of images for batch inference")
    parser.add_argument("--top_k",     type=int, default=3,
                        help="Number of top predictions to return")
    args = parser.parse_args()

    print(f"\n📦  Loading model: {args.model}")
    model       = load_model(args.model)
    class_names = load_class_names(args.train_dir)
    print(f"🏷️   Classes ({len(class_names)}): {class_names}\n")

    if args.image:
        result = predict_single(model, args.image, class_names, top_k=args.top_k)
        print(f"📍  Predicted Landmark : {result['prediction']}")
        print(f"    Confidence         : {result['confidence']}")
        print(f"\n    Top-{args.top_k} Predictions:")
        for item in result['top_k']:
            print(f"      {item['rank']}. {item['landmark']:30s}  {item['confidence']}")
        save_results([result])

    elif args.folder:
        print(f"🗂️   Running batch inference on: {args.folder}\n")
        results = predict_batch(model, args.folder, class_names)
        save_results(results)
        correct = sum(1 for r in results if Path(r['image']).parent.name == r['prediction'])
        print(f"\n📊  Batch Accuracy: {correct}/{len(results)} = "
              f"{correct / len(results) * 100:.1f}%")

    else:
        print("❌  Please provide --image or --folder")
