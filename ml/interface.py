"""
Smart Waste Segregation AI
Model Interface

Loads the trained EfficientNet-B0 model and provides
a simple interface for image prediction.

Usage:
    python ml/interface.py path/to/image.jpg

Or from Python:
    from ml.interface import predict_image
    result = predict_image("image.jpg")
"""

from pathlib import Path
import json
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "training_outputs"
    / "efficientnet_b0_cached"
)

MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
CLASS_TO_IDX_PATH = MODEL_DIR / "class_to_idx.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

# Must match the 224x224 input used during training.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():
    """
    Load class names from class_names.json.

    Supports either:
        ["Battery", "Cardboard", ...]
    or:
        {"0": "Battery", "1": "Cardboard", ...}
    """

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Handle {"0": "Battery", ...}
        return [
            data[str(i)] if str(i) in data else data[i]
            for i in range(len(data))
        ]

    raise ValueError(
        "Unsupported class_names.json format."
    )


# ============================================================
# LOAD CLASS-TO-INDEX
# ============================================================

def load_class_to_idx():
    """
    Load class-to-index mapping if available.
    """

    if not CLASS_TO_IDX_PATH.exists():
        return None

    with open(CLASS_TO_IDX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(num_classes):
    """
    Create EfficientNet-B0 architecture matching training.
    """

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = torch.nn.Linear(
        in_features,
        num_classes
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained EfficientNet-B0 model.
    """

    class_names = load_class_names()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model = create_model(len(class_names))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # --------------------------------------------------------
    # Support several checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        else:
            # Could already be a raw state_dict
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove possible "module." prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model, class_names


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

_MODEL = None
_CLASS_NAMES = None


def get_model():
    """
    Load the model only once.

    This is important for the application because we don't
    want to reload the 128 MB model for every image.
    """

    global _MODEL
    global _CLASS_NAMES

    if _MODEL is None:
        _MODEL, _CLASS_NAMES = load_model()

    return _MODEL, _CLASS_NAMES


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(
    image_path,
    top_k=3
):
    """
    Predict waste class from an image.

    Parameters
    ----------
    image_path : str or Path
        Path to input image.

    top_k : int
        Number of top predictions to return.

    Returns
    -------
    dict
        Prediction result.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    model, class_names = get_model()

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(
            f"Could not read image:\n{image_path}\n\n{e}"
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    tensor = TRANSFORM(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    # --------------------------------------------------------
    # Top-K predictions
    # --------------------------------------------------------

    top_k = min(
        top_k,
        len(class_names)
    )

    values, indices = torch.topk(
        probabilities,
        k=top_k
    )

    predictions = []

    for probability, index in zip(
        values,
        indices
    ):

        class_index = int(index.item())

        predictions.append({
            "class_id": class_index,
            "class_name": class_names[class_index],
            "confidence": float(probability.item()),
            "confidence_percent": round(
                float(probability.item()) * 100,
                2
            )
        })

    # --------------------------------------------------------
    # Main prediction
    # --------------------------------------------------------

    best = predictions[0]

    result = {
        "image": str(image_path),
        "predicted_class": best["class_name"],
        "class_id": best["class_id"],
        "confidence": best["confidence"],
        "confidence_percent": best["confidence_percent"],
        "top_predictions": predictions,
        "device": str(DEVICE),
        "model": "efficientnet_b0",
    }

    return result


# ============================================================
# PRINT RESULT
# ============================================================

def print_prediction(result):

    print()
    print("=" * 60)
    print("SMART WASTE SEGREGATION AI")
    print("=" * 60)

    print(f"Image      : {result['image']}")
    print(f"Model      : {result['model']}")
    print(f"Device     : {result['device']}")

    print()
    print("PREDICTION")
    print("-" * 60)

    print(
        f"Class      : {result['predicted_class']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence_percent']:.2f}%"
    )

    print()
    print("TOP PREDICTIONS")
    print("-" * 60)

    for i, prediction in enumerate(
        result["top_predictions"],
        start=1
    ):

        print(
            f"{i}. "
            f"{prediction['class_name']:<30} "
            f"{prediction['confidence_percent']:>6.2f}%"
        )

    print("=" * 60)
    print()


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            "  python ml/interface.py "
            "path/to/image.jpg"
        )
        print()

        return

    image_path = sys.argv[1]

    try:

        result = predict_image(
            image_path,
            top_k=3
        )

        print_prediction(result)

    except Exception as e:

        print()
        print("ERROR")
        print("-" * 60)
        print(str(e))
        print()

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()