"""
SMART WASTE SEGREGATION AI
WHY THIS RESULT — GRAD-CAM EXPLANATION

Uses the trained EfficientNet-B0 model to:
1. Load a waste image
2. Predict the waste class
3. Generate Grad-CAM heatmap
4. Overlay the heatmap on the original image
5. Save the explanation image

Usage:

    python ml/explain.py "data/raw/realwaste/Plastic/Plastic_821.jpg"

Optional:

    python ml/explain.py \
        "data/raw/realwaste/Plastic/Plastic_821.jpg" \
        --output training_outputs/explanations
"""

from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "training_outputs"
    / "efficientnet_b0_cached"
    / "best_model.pth"
)

DEFAULT_CLASS_NAMES_PATH = (
    PROJECT_ROOT
    / "training_outputs"
    / "efficientnet_b0_cached"
    / "class_names.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "training_outputs"
    / "explanations"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


# ============================================================
# MODEL
# ============================================================

def build_model(num_classes):
    """
    Build EfficientNet-B0 with the same classifier
    structure used during training.
    """

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    return model


# ============================================================
# CLASS MAPPING
# ============================================================

def extract_class_names_from_mapping(mapping):
    """
    Convert different possible class mapping formats
    into:

        ["Battery", "Cardboard", ...]
    """

    if mapping is None:
        return None

    # --------------------------------------------------------
    # Dictionary:
    #
    # {"0": "Battery", "1": "Cardboard"}
    # --------------------------------------------------------

    if isinstance(mapping, dict):

        result = []

        for i in range(len(mapping)):

            if str(i) in mapping:
                result.append(
                    str(mapping[str(i)])
                )

            elif i in mapping:
                result.append(
                    str(mapping[i])
                )

            else:
                return None

        return result

    # --------------------------------------------------------
    # List:
    #
    # ["Battery", "Cardboard", ...]
    # --------------------------------------------------------

    if isinstance(mapping, (list, tuple)):

        return [
            str(x)
            for x in mapping
        ]

    return None


def load_class_names(
    checkpoint,
    class_names_path,
):
    """
    Robust class-name loading.

    Priority:

    1. checkpoint["idx_to_class"]
    2. checkpoint["class_names"]
    3. class_names.json["classes"]
    4. class_names.json["idx_to_class"]
    5. class_names.json["class_to_idx"]
    6. direct numeric mapping
    """

    # --------------------------------------------------------
    # 1. CHECKPOINT idx_to_class
    # --------------------------------------------------------

    if "idx_to_class" in checkpoint:

        names = extract_class_names_from_mapping(
            checkpoint["idx_to_class"]
        )

        if names:
            return names

    # --------------------------------------------------------
    # 2. CHECKPOINT class_names
    # --------------------------------------------------------

    if "class_names" in checkpoint:

        names = extract_class_names_from_mapping(
            checkpoint["class_names"]
        )

        if names:
            return names

    # --------------------------------------------------------
    # 3. JSON FILE
    # --------------------------------------------------------

    if class_names_path.exists():

        with open(
            class_names_path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        # {"classes": [...]}

        if "classes" in data:

            names = extract_class_names_from_mapping(
                data["classes"]
            )

            if names:
                return names

        # {"idx_to_class": {...}}

        if "idx_to_class" in data:

            names = extract_class_names_from_mapping(
                data["idx_to_class"]
            )

            if names:
                return names

        # {"class_to_idx": {...}}

        if "class_to_idx" in data:

            class_to_idx = data["class_to_idx"]

            if isinstance(
                class_to_idx,
                dict,
            ):

                result = [
                    None
                ] * len(class_to_idx)

                for class_name, index in (
                    class_to_idx.items()
                ):

                    result[int(index)] = (
                        str(class_name)
                    )

                if all(
                    x is not None
                    for x in result
                ):
                    return result

        # Direct numeric mapping

        names = extract_class_names_from_mapping(
            data
        )

        if names:
            return names

    raise RuntimeError(
        "\nCould not determine class names.\n"
        "Checked:\n"
        f"  {class_names_path}\n"
        "and the model checkpoint."
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path=DEFAULT_MODEL_PATH,
    class_names_path=DEFAULT_CLASS_NAMES_PATH,
    device=None,
):

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model_path = Path(model_path)
    class_names_path = Path(
        class_names_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "\nModel checkpoint not found:\n"
            f"{model_path}\n"
        )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------------------
    # GET NUMBER OF CLASSES
    # --------------------------------------------------------

    num_classes = checkpoint.get(
        "num_classes"
    )

    if num_classes is None:

        if "idx_to_class" in checkpoint:

            num_classes = len(
                checkpoint["idx_to_class"]
            )

        elif "class_names" in checkpoint:

            num_classes = len(
                checkpoint["class_names"]
            )

        else:

            num_classes = 16

    num_classes = int(
        num_classes
    )

    # --------------------------------------------------------
    # CLASS NAMES
    # --------------------------------------------------------

    class_names = load_class_names(
        checkpoint,
        class_names_path,
    )

    if len(class_names) != num_classes:

        raise RuntimeError(
            "\nClass mapping mismatch.\n"
            f"Model classes : {num_classes}\n"
            f"Class names   : {len(class_names)}\n"
        )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_model(
        num_classes
    )

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        raise RuntimeError(
            "Model checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        class_names,
        checkpoint,
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Load and preprocess image for EfficientNet-B0.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"\nImage not found:\n"
            f"{image_path}\n"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )

    tensor = transform(
        image
    ).unsqueeze(0)

    return (
        image,
        tensor,
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    """
    Simple Grad-CAM implementation.

    No pytorch-grad-cam package required.
    """

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = (
            target_layer
        )

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = (
            output.detach()
        )

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = (
            grad_output[0].detach()
        )

    def generate(
        self,
        image_tensor,
        class_index,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            image_tensor
        )

        score = output[
            0,
            class_index
        ]

        score.backward()

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # ----------------------------------------------------
        # Weighted activation maps
        # ----------------------------------------------------

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=False,
        )

        cam = torch.relu(
            cam
        )

        cam = cam[0]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:

            cam /= max_value

        return (
            cam.cpu().numpy()
        )

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# HEATMAP
# ============================================================

def create_heatmap(
    cam,
    width,
    height,
):
    """
    Resize Grad-CAM to original image size
    and create an RGB heatmap.
    """

    cam_image = Image.fromarray(
        np.uint8(
            cam * 255
        )
    )

    cam_image = cam_image.resize(
        (
            width,
            height,
        ),
        Image.Resampling.BILINEAR,
    )

    cam_array = np.asarray(
        cam_image
    )

    # --------------------------------------------------------
    # Simple JET-style heatmap
    # --------------------------------------------------------

    heatmap = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    normalized = (
        cam_array.astype(
            np.float32
        )
        / 255.0
    )

    # Blue -> Cyan -> Yellow -> Red

    heatmap[:, :, 0] = np.uint8(
        np.clip(
            255
            * (2.0 * normalized - 0.25),
            0,
            255,
        )
    )

    heatmap[:, :, 1] = np.uint8(
        np.clip(
            255
            * (
                1.0
                - np.abs(
                    2.0
                    * normalized
                    - 1.0
                )
            ),
            0,
            255,
        )
    )

    heatmap[:, :, 2] = np.uint8(
        np.clip(
            255
            * (
                1.0
                - 2.0
                * normalized
            ),
            0,
            255,
        )
    )

    return Image.fromarray(
        heatmap
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    original_image,
    heatmap,
    alpha=0.45,
):

    original = original_image.convert(
        "RGB"
    )

    heatmap = heatmap.convert(
        "RGB"
    )

    return Image.blend(
        original,
        heatmap,
        alpha,
    )


# ============================================================
# EXPLANATION
# ============================================================

def generate_text_explanation(
    predicted_class,
    confidence,
):

    confidence_percent = (
        confidence * 100
    )

    if confidence >= 0.90:

        confidence_text = (
            "The model is highly confident "
            "in this prediction."
        )

    elif confidence >= 0.70:

        confidence_text = (
            "The model has reasonably strong "
            "confidence in this prediction."
        )

    elif confidence >= 0.50:

        confidence_text = (
            "The model has moderate confidence, "
            "so this result should be treated "
            "with some caution."
        )

    else:

        confidence_text = (
            "The model has low confidence, "
            "so the image may need manual review."
        )

    explanation = (
        f"The model classified the image as "
        f"'{predicted_class}' with "
        f"{confidence_percent:.2f}% confidence. "
        f"{confidence_text} "
        "The Grad-CAM heatmap highlights the "
        "regions of the image that contributed "
        "most strongly to this prediction."
    )

    return explanation


# ============================================================
# MAIN EXPLANATION FUNCTION
# ============================================================

def explain_image(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):

    image_path = Path(
        image_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, class_names, checkpoint = (
        load_model(
            model_path=model_path,
            device=device,
        )
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    original_image, image_tensor = (
        preprocess_image(
            image_path
        )
    )

    image_tensor = image_tensor.to(
        device
    )

    # --------------------------------------------------------
    # FIND EFFICIENTNET TARGET LAYER
    # --------------------------------------------------------

    # EfficientNet-B0 feature extractor.
    #
    # The final convolutional feature block is:
    #
    # model.features[-1]

    target_layer = (
        model.features[-1]
    )

    gradcam = GradCAM(
        model,
        target_layer,
    )

    try:

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, predicted_index = (
            probabilities.max(
                dim=1
            )
        )

        predicted_index = int(
            predicted_index.item()
        )

        confidence = float(
            confidence.item()
        )

        predicted_class = (
            class_names[
                predicted_index
            ]
        )

        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        top_k = min(
            3,
            len(class_names)
        )

        top_probs, top_indices = (
            probabilities[0].topk(
                top_k
            )
        )

        top_predictions = []

        for prob, index in zip(
            top_probs,
            top_indices,
        ):

            index = int(
                index.item()
            )

            top_predictions.append(
                {
                    "class":
                        class_names[index],

                    "confidence":
                        float(
                            prob.item()
                        ),
                }
            )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        cam = gradcam.generate(
            image_tensor,
            predicted_index,
        )

    finally:

        gradcam.close()

    # --------------------------------------------------------
    # CREATE VISUALIZATION
    # --------------------------------------------------------

    heatmap = create_heatmap(
        cam,
        original_image.width,
        original_image.height,
    )

    overlay = create_overlay(
        original_image,
        heatmap,
    )

    # --------------------------------------------------------
    # OUTPUT NAME
    # --------------------------------------------------------

    safe_name = (
        image_path.stem
        + "_why_result.jpg"
    )

    output_path = (
        output_dir
        / safe_name
    )

    overlay.save(
        output_path,
        quality=95,
    )

    # --------------------------------------------------------
    # TEXT EXPLANATION
    # --------------------------------------------------------

    explanation = (
        generate_text_explanation(
            predicted_class,
            confidence,
        )
    )

    return {
        "image":
            str(image_path),

        "prediction":
            predicted_class,

        "class_index":
            predicted_index,

        "confidence":
            confidence,

        "confidence_percent":
            confidence * 100,

        "top_predictions":
            top_predictions,

        "explanation":
            explanation,

        "explanation_image":
            str(output_path),

        "device":
            str(device),

        "architecture":
            "efficientnet_b0",
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

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
        f"{result['confidence_percent']:.2f}%"
    )

    print()
    print("TOP PREDICTIONS")
    print("-" * 60)

    for index, item in enumerate(
        result["top_predictions"],
        start=1,
    ):

        print(
            f"{index}. "
            f"{item['class']:<30}"
            f"{item['confidence'] * 100:>6.2f}%"
        )

    print()
    print("EXPLANATION")
    print("-" * 60)

    print(
        result["explanation"]
    )

    print()
    print(
        "Grad-CAM explanation:"
    )

    print(
        result["explanation_image"]
    )

    print()
    print("=" * 60)


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM explanation "
            "for waste classification."
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Path to input waste image"
        ),
    )

    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL_PATH
        ),
        help=(
            "Path to trained model checkpoint"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Directory for explanation images"
        ),
    )

    args = parser.parse_args()

    try:

        result = explain_image(
            image_path=args.image,
            model_path=args.model,
            output_dir=args.output,
        )

        print_result(
            result
        )

    except Exception as exc:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(
            str(exc)
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()