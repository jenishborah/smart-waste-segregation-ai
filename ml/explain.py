"""
Smart Waste Segregation AI
WHY THIS RESULT — Grad-CAM Explanation

Usage:

    python ml/explain.py "data/raw/realwaste/Plastic/Plastic_821.jpg"

Output:

    explanation_<image>.jpg

The generated image shows:
    - Original image
    - Grad-CAM heatmap
    - Overlay showing important regions
"""

from pathlib import Path
import sys
import json

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "training_outputs"
    / "efficientnet_b0_cached"
)

MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training_outputs"
    / "explanations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD CLASSES
# ============================================================

def load_class_names():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        return [
            data[str(i)]
            for i in range(len(data))
        ]

    raise ValueError(
        "Unsupported class_names.json format"
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(num_classes):

    model = models.efficientnet_b0(
        weights=None
    )

    in_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = torch.nn.Linear(
        in_features,
        num_classes
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    class_names = load_class_names()

    model = create_model(
        len(class_names)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True
    )

    model.to(DEVICE)

    return model, class_names


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        image_tensor,
        class_index
    ):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        score = output[
            0,
            class_index
        ]

        score.backward()

        activations = self.activations
        gradients = self.gradients

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.detach().cpu().numpy()


# ============================================================
# GENERATE EXPLANATION
# ============================================================

def explain_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model, class_names = load_model()

    model.eval()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original = Image.open(
        image_path
    ).convert("RGB")

    original_resized = original.resize(
        (224, 224)
    )

    image_tensor = TRANSFORM(
        original
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # EfficientNet target layer
    # --------------------------------------------------------

    target_layer = (
        model.features[-1]
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    output = model(
        image_tensor
    )

    probabilities = F.softmax(
        output,
        dim=1
    )[0]

    confidence, prediction = (
        probabilities.max(dim=0)
    )

    class_index = int(
        prediction.item()
    )

    class_name = class_names[
        class_index
    ]

    confidence_percent = (
        float(confidence.item()) * 100
    )

    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    cam = gradcam.generate(
        image_tensor,
        class_index
    )

    # --------------------------------------------------------
    # Save explanation image
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"explanation_{image_path.stem}.jpg"
    )

    original_array = np.asarray(
        original_resized
    ) / 255.0

    plt.figure(
        figsize=(15, 5)
    )

    # Original
    plt.subplot(1, 3, 1)

    plt.imshow(
        original_array
    )

    plt.title(
        "Input Image"
    )

    plt.axis("off")

    # Heatmap
    plt.subplot(1, 3, 2)

    plt.imshow(
        cam,
        cmap="jet"
    )

    plt.title(
        "Why This Result?"
    )

    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)

    plt.imshow(
        original_array
    )

    plt.imshow(
        cam,
        cmap="jet",
        alpha=0.45
    )

    plt.title(
        f"{class_name} "
        f"({confidence_percent:.1f}%)"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "image": str(image_path),
        "prediction": class_name,
        "class_id": class_index,
        "confidence": confidence_percent,
        "explanation_image": str(
            output_path
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            'python ml/explain.py '
            '"path/to/image.jpg"'
        )

        return

    image_path = sys.argv[1]

    try:

        result = explain_image(
            image_path
        )

        print()
        print("=" * 60)
        print("WHY THIS RESULT")
        print("=" * 60)

        print(
            f"Prediction : "
            f"{result['prediction']}"
        )

        print(
            f"Confidence : "
            f"{result['confidence']:.2f}%"
        )

        print(
            f"Explanation:"
        )

        print(
            result["explanation_image"]
        )

        print("=" * 60)

    except Exception as e:

        print()
        print("ERROR")
        print("-" * 60)
        print(str(e))

        raise


if __name__ == "__main__":
    main()