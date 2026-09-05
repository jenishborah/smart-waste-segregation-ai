"""Local EfficientNet-B0 inference interface.

Usage:
    python ml/interface.py "data/raw/realwaste/Plastic/Plastic_821.jpg"
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "training_outputs" / "efficientnet_b0_cached"
MODEL_PATH = MODEL_DIR / "best_model.pth"


def normalize_mapping(data):
    if isinstance(data, list):
        return {i: str(v) for i, v in enumerate(data)}
    if not isinstance(data, dict):
        return None

    if isinstance(data.get("classes"), list):
        return {i: str(v) for i, v in enumerate(data["classes"])}

    if isinstance(data.get("idx_to_class"), dict):
        return {int(k): str(v) for k, v in data["idx_to_class"].items()}

    if isinstance(data.get("class_to_idx"), dict):
        return {int(v): str(k) for k, v in data["class_to_idx"].items()}

    if all(str(k).isdigit() for k in data):
        return {int(k): str(v) for k, v in data.items()}

    return None


def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


def load_class_names(checkpoint):
    # The checkpoint created by the training pipeline contains idx_to_class.
    for key in ("idx_to_class", "class_names"):
        if key in checkpoint:
            mapping = normalize_mapping(checkpoint[key])
            if mapping:
                return mapping

    candidates = [
        MODEL_DIR / "class_names.json",
        ROOT / "models" / "class_names.json",
        MODEL_DIR / "class_to_idx.json",
        ROOT / "models" / "class_to_idx.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                mapping = normalize_mapping(json.load(f))
            if mapping:
                return mapping

    raise FileNotFoundError(
        "No usable class mapping found. Checked checkpoint and: "
        + ", ".join(str(p) for p in candidates)
    )


def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    architecture = str(
        checkpoint.get("architecture", "efficientnet_b0")
    ).lower()

    if architecture != "efficientnet_b0":
        raise ValueError(
            f"Expected efficientnet_b0, got {architecture}"
        )

    num_classes = int(checkpoint.get("num_classes", 16))
    image_size = int(checkpoint.get("image_size", 224))

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = load_class_names(checkpoint)

    return model, class_names, image_size


def predict(model, image_path, class_names, image_size, device, top_k):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tensor = transform(image).unsqueeze(0).to(
        device, non_blocking=True
    )

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    k = min(max(1, top_k), probabilities.numel())
    values, indices = torch.topk(probabilities, k=k)

    results = []
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        index = int(index)
        results.append((
            class_names.get(index, f"Unknown Class {index}"),
            float(value),
        ))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the trained EfficientNet-B0 waste classifier."
    )
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    try:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model, class_names, image_size = load_model(device)

        results = predict(
            model,
            args.image,
            class_names,
            image_size,
            device,
            args.top_k,
        )

        best_class, best_confidence = results[0]

        print()
        print("=" * 60)
        print("SMART WASTE SEGREGATION AI")
        print("=" * 60)
        print(f"Image      : {args.image}")
        print("Model      : efficientnet_b0")
        print(f"Device     : {device}")

        if device.type == "cuda":
            print(f"GPU        : {torch.cuda.get_device_name(0)}")

        print()
        print("PREDICTION")
        print("-" * 60)
        print(f"Class      : {best_class}")
        print(f"Confidence : {best_confidence * 100:.2f}%")

        print()
        print("TOP PREDICTIONS")
        print("-" * 60)

        for rank, (name, confidence) in enumerate(results, 1):
            print(f"{rank}. {name:<30} {confidence * 100:6.2f}%")

        print("=" * 60)

    except Exception as exc:
        import traceback
        print()
        print("ERROR")
        print("-" * 60)
        print(f"{type(exc).__name__}: {exc}")
        print("-" * 60)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
