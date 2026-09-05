"""
SMART WASTE SEGREGATION AI
UNIFIED INFERENCE PIPELINE

Pipeline:
    Image
      ↓
    Image Quality Check
      ↓
    EfficientNet-B0
      ↓
    Prediction + Confidence
      ↓
    Top-3 Predictions
      ↓
    Why This Result? / Grad-CAM
      ↓
    Recycling Guidance
      ↓
    Structured JSON

Usage:
    python ml/pipeline.py "data/raw/realwaste/Plastic/Plastic_821.jpg"
"""

from pathlib import Path
import argparse
import json
import subprocess
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFile


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "training_outputs"
    / "efficientnet_b0_cached"
    / "best_model.pth"
)

CLASS_NAMES_PATH = (
    PROJECT_ROOT
    / "training_outputs"
    / "efficientnet_b0_cached"
    / "class_names.json"
)

EXPLAIN_SCRIPT = (
    PROJECT_ROOT
    / "ml"
    / "explain.py"
)

EXPLANATION_DIR = (
    PROJECT_ROOT
    / "training_outputs"
    / "explanations"
)

PIPELINE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "training_outputs"
    / "pipeline_results"
)

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

ImageFile.LOAD_TRUNCATED_IMAGES = False


# ============================================================
# CLASS NAMES
# ============================================================

DEFAULT_CLASSES = [
    "Battery",
    "Cardboard",
    "Electronic Component",
    "Electronic Device",
    "Food Organics",
    "Glass",
    "Hazardous Waste",
    "Large Electronic Appliance",
    "Metal",
    "Organic Stream",
    "Paper",
    "Plastic",
    "Recyclable Stream",
    "Residual",
    "Textile",
    "Vegetation",
]


# ============================================================
# RECYCLING / DISPOSAL GUIDANCE
# ============================================================

RECYCLING_GUIDANCE = {

    "Battery": (
        "Do not place batteries in regular household waste. "
        "Keep them separate and take them to an appropriate "
        "battery collection or e-waste facility."
    ),

    "Cardboard": (
        "Keep cardboard clean and dry. Flatten boxes and place "
        "them in the appropriate paper/cardboard recycling stream."
    ),

    "Electronic Component": (
        "Do not place electronic components in regular waste. "
        "Keep them separate and send them to an authorized "
        "e-waste collection facility."
    ),

    "Electronic Device": (
        "Do not dispose of electronic devices with regular waste. "
        "Use an authorized e-waste collection or recycling facility."
    ),

    "Food Organics": (
        "Place food waste in the appropriate organic-waste or "
        "composting stream. Avoid mixing it with recyclable materials."
    ),

    "Glass": (
        "Empty and rinse the glass item when appropriate, then "
        "place it in the designated glass recycling stream. "
        "Handle broken glass carefully."
    ),

    "Hazardous Waste": (
        "Do not place hazardous waste in regular recycling or "
        "household waste. Keep it separate and use an appropriate "
        "hazardous-waste collection facility."
    ),

    "Large Electronic Appliance": (
        "Do not place large electronic appliances in regular waste. "
        "Use an authorized e-waste or appliance recycling service."
    ),

    "Metal": (
        "Separate metal from general waste. Clean the item when "
        "appropriate and place it in the designated metal recycling stream."
    ),

    "Organic Stream": (
        "Place suitable organic material in the designated organic "
        "waste or composting stream."
    ),

    "Paper": (
        "Keep paper clean and dry and place it in the appropriate "
        "paper recycling stream."
    ),

    "Plastic": (
        "Empty and rinse the plastic item when appropriate, then "
        "place it in the designated plastic recycling stream. "
        "Follow your local recycling rules."
    ),

    "Recyclable Stream": (
        "This item appears suitable for a recyclable stream. "
        "Keep it clean and dry and follow your local recycling rules."
    ),

    "Residual": (
        "This item appears to belong to residual/general waste. "
        "Do not place it with recyclable materials unless local "
        "guidelines specifically allow it."
    ),

    "Textile": (
        "Keep textiles separate from regular waste where possible. "
        "Reuse, donate, or place them in an appropriate textile "
        "collection stream."
    ),

    "Vegetation": (
        "Place vegetation and garden waste in the appropriate "
        "organic-waste or composting stream."
    ),
}


# ============================================================
# DEVICE
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names(checkpoint=None):

    # --------------------------------------------------------
    # Checkpoint mapping
    # --------------------------------------------------------

    if checkpoint is not None:

        idx_to_class = checkpoint.get(
            "idx_to_class"
        )

        if isinstance(idx_to_class, dict):

            try:

                return [
                    idx_to_class[str(i)]
                    for i in range(len(idx_to_class))
                ]

            except (KeyError, TypeError):
                pass

        class_to_idx = checkpoint.get(
            "class_to_idx"
        )

        if isinstance(class_to_idx, dict):

            try:

                ordered = sorted(
                    class_to_idx.items(),
                    key=lambda item: item[1],
                )

                return [
                    name
                    for name, _ in ordered
                ]

            except Exception:
                pass

    # --------------------------------------------------------
    # class_names.json
    # --------------------------------------------------------

    if CLASS_NAMES_PATH.exists():

        try:

            with open(
                CLASS_NAMES_PATH,
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(f)

            if isinstance(data, dict):

                classes = data.get("classes")

                if isinstance(classes, list):

                    return classes

                numeric_keys = all(
                    str(i) in data
                    for i in range(len(data))
                )

                if numeric_keys:

                    return [
                        data[str(i)]
                        for i in range(len(data))
                    ]

        except Exception as exc:

            print(
                f"WARNING: Could not read "
                f"class_names.json: {exc}"
            )

    return DEFAULT_CLASSES.copy()


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(num_classes):

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nModel not found:\n"
            f"{MODEL_PATH}\n\n"
            "Make sure best_model.pth exists."
        )

    device = get_device()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    num_classes = int(
        checkpoint.get(
            "num_classes",
            len(DEFAULT_CLASSES),
        )
    )

    model = build_model(
        num_classes
    )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)

    model.eval()

    class_names = load_class_names(
        checkpoint
    )

    if len(class_names) != num_classes:

        raise RuntimeError(
            "Class count mismatch.\n"
            f"Model expects: {num_classes}\n"
            f"Class names:   {len(class_names)}"
        )

    return (
        model,
        class_names,
        device,
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_path):

    path = Path(image_path)

    if not path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Not a file:\n{path}"
        )

    try:

        with Image.open(path) as image:

            image.load()

            width, height = image.size

            if width < 64 or height < 64:

                return {
                    "status": "WARNING",
                    "message": (
                        "Image resolution is very small."
                    ),
                    "width": width,
                    "height": height,
                }

            return {
                "status": "OK",
                "message": "Image is readable.",
                "width": width,
                "height": height,
            }

    except Exception as exc:

        raise RuntimeError(
            "Image could not be read:\n"
            f"{path}\n\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc


# ============================================================
# TRANSFORM
# ============================================================

def get_transform():

    return transforms.Compose([

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
    ])


# ============================================================
# PREDICTION
# ============================================================

@torch.inference_mode()
def predict(
    model,
    image_path,
    class_names,
    device,
    top_k=3,
):

    transform = get_transform()

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        tensor = transform(
            image
        )

    tensor = tensor.unsqueeze(
        0
    ).to(device)

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )[0]

    k = min(
        top_k,
        len(class_names),
    )

    values, indices = torch.topk(
        probabilities,
        k=k,
    )

    predictions = []

    for value, index in zip(
        values,
        indices,
    ):

        class_index = int(
            index.item()
        )

        confidence = float(
            value.item()
        )

        predictions.append({

            "class": class_names[
                class_index
            ],

            "confidence": confidence,

            "confidence_percent":
                round(
                    confidence * 100,
                    2,
                ),

            "class_index":
                class_index,
        })

    best = predictions[0]

    return {
        "prediction": best["class"],
        "confidence": best["confidence"],
        "confidence_percent":
            best["confidence_percent"],
        "top_predictions":
            predictions,
    }


# ============================================================
# WHY THIS RESULT
# ============================================================

def generate_why_result(
    image_path,
):

    """
    Calls the existing explain.py.

    This keeps Grad-CAM implementation in one place
    instead of duplicating it inside pipeline.py.
    """

    if not EXPLAIN_SCRIPT.exists():

        return {
            "status": "UNAVAILABLE",
            "message": (
                "explain.py was not found."
            ),
            "path": None,
        }

    EXPLANATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        completed = subprocess.run(
            [
                sys.executable,
                str(EXPLAIN_SCRIPT),
                str(image_path),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode != 0:

            return {
                "status": "ERROR",
                "message": (
                    "Grad-CAM explanation failed."
                ),
                "path": None,
                "error": stderr.strip(),
            }

        # ----------------------------------------------------
        # Find generated explanation path
        # ----------------------------------------------------

        explanation_path = None

        for line in stdout.splitlines():

            line = line.strip()

            if (
                "Grad-CAM explanation:"
                in line
            ):

                explanation_path = line.split(
                    "Grad-CAM explanation:",
                    1,
                )[1].strip()

                break

            if (
                "Explanation:"
                in line
            ):

                candidate = line.split(
                    "Explanation:",
                    1,
                )[1].strip()

                if candidate:

                    explanation_path = candidate

        # ----------------------------------------------------
        # Fallback: search expected output
        # ----------------------------------------------------

        if explanation_path is None:

            candidates = sorted(
                EXPLANATION_DIR.glob(
                    f"{Path(image_path).stem}*"
                ),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            if candidates:

                explanation_path = str(
                    candidates[0]
                )

        if explanation_path:

            path_obj = Path(
                explanation_path
            )

            if path_obj.exists():

                return {
                    "status": "SUCCESS",
                    "message": (
                        "Grad-CAM explanation generated."
                    ),
                    "path": str(
                        path_obj.resolve()
                    ),
                }

        return {
            "status": "SUCCESS",
            "message": (
                "Grad-CAM completed, but "
                "output path could not be detected."
            ),
            "path": None,
        }

    except subprocess.TimeoutExpired:

        return {
            "status": "TIMEOUT",
            "message": (
                "Grad-CAM explanation timed out."
            ),
            "path": None,
        }

    except Exception as exc:

        return {
            "status": "ERROR",
            "message": (
                "Could not run explain.py."
            ),
            "path": None,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


# ============================================================
# EXPLANATION TEXT
# ============================================================

def create_explanation_text(
    prediction,
    confidence_percent,
):

    if confidence_percent >= 90:

        confidence_text = (
            "The model is highly confident in this prediction."
        )

    elif confidence_percent >= 70:

        confidence_text = (
            "The model has relatively high confidence "
            "in this prediction."
        )

    elif confidence_percent >= 50:

        confidence_text = (
            "The model has moderate confidence, so the "
            "prediction should be interpreted with some caution."
        )

    else:

        confidence_text = (
            "The model has low confidence, so the result "
            "should be verified before disposal."
        )

    return (
        f"The model classified the image as "
        f"'{prediction}' with "
        f"{confidence_percent:.2f}% confidence. "
        f"{confidence_text} "
        f"The Grad-CAM heatmap highlights regions of "
        f"the image that contributed most strongly "
        f"to the prediction."
    )


# ============================================================
# RECYCLING GUIDANCE
# ============================================================

def get_recycling_guidance(
    prediction,
):

    return RECYCLING_GUIDANCE.get(
        prediction,
        (
            "Follow your local waste-management guidelines "
            "for this material."
        ),
    )


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    image_path,
    result,
    quality,
    explanation,
    guidance,
    device,
):

    print()
    print("=" * 60)
    print(
        "SMART WASTE SEGREGATION AI"
    )
    print("=" * 60)

    print(
        f"Image      : {image_path}"
    )

    print(
        "Model      : efficientnet_b0"
    )

    print(
        f"Device     : {device}"
    )

    print()
    print("IMAGE QUALITY")
    print("-" * 60)

    print(
        f"Status     : {quality['status']}"
    )

    print(
        f"Resolution : "
        f"{quality['width']} x "
        f"{quality['height']}"
    )

    print()
    print("PREDICTION")
    print("-" * 60)

    print(
        f"Class      : "
        f"{result['prediction']}"
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
        start=1,
    ):

        print(
            f"{i}. "
            f"{prediction['class']:<32}"
            f"{prediction['confidence_percent']:>7.2f}%"
        )

    print()
    print("WHY THIS RESULT?")
    print("-" * 60)

    print(
        create_explanation_text(
            result["prediction"],
            result["confidence_percent"],
        )
    )

    print()
    print("GRAD-CAM")
    print("-" * 60)

    if explanation["status"] == "SUCCESS":

        if explanation["path"]:

            print(
                f"Explanation : "
                f"{explanation['path']}"
            )

        else:

            print(
                "Grad-CAM explanation generated."
            )

    else:

        print(
            f"Status      : "
            f"{explanation['status']}"
        )

        print(
            f"Message     : "
            f"{explanation['message']}"
        )

    print()
    print("RECYCLING GUIDANCE")
    print("-" * 60)

    print(guidance)

    print("=" * 60)


# ============================================================
# SAVE JSON
# ============================================================

def save_result(
    image_path,
    result,
    quality,
    explanation,
    guidance,
    device,
):

    prediction = result["prediction"]

    explanation_text = create_explanation_text(
        prediction,
        result["confidence_percent"],
    )

    output = {

        "image":
            str(
                Path(image_path).resolve()
            ),

        "model":
            "efficientnet_b0",

        "device":
            str(device),

        "quality":
            quality,

        "prediction":
            prediction,

        "confidence":
            result["confidence"],

        "confidence_percent":
            result["confidence_percent"],

        "top_predictions":
            result["top_predictions"],

        "why_this_result":
            explanation_text,

        "grad_cam":
            explanation,

        "recycling_guidance":
            guidance,
    }

    PIPELINE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PIPELINE_OUTPUT_DIR
        / (
            Path(image_path).stem
            + "_result.json"
        )
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Unified Smart Waste Segregation "
            "inference pipeline"
        )
    )

    parser.add_argument(
        "image",
        help="Path to input image",
    )

    parser.add_argument(
        "--no-explain",
        action="store_true",
        help=(
            "Skip Grad-CAM generation "
            "for faster inference"
        ),
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    try:

        # ----------------------------------------------------
        # Image quality
        # ----------------------------------------------------

        quality = validate_image(
            image_path
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model, class_names, device = (
            load_model()
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = predict(
            model=model,
            image_path=image_path,
            class_names=class_names,
            device=device,
            top_k=3,
        )

        # ----------------------------------------------------
        # Why this result
        # ----------------------------------------------------

        if args.no_explain:

            explanation = {
                "status": "SKIPPED",
                "message": (
                    "Grad-CAM skipped by user."
                ),
                "path": None,
            }

        else:

            print()
            print(
                "Generating Grad-CAM explanation..."
            )

            explanation = generate_why_result(
                image_path
            )

        # ----------------------------------------------------
        # Recycling guidance
        # ----------------------------------------------------

        guidance = get_recycling_guidance(
            result["prediction"]
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print_result(
            image_path=image_path,
            result=result,
            quality=quality,
            explanation=explanation,
            guidance=guidance,
            device=device,
        )

        # ----------------------------------------------------
        # Save JSON
        # ----------------------------------------------------

        output_path = save_result(
            image_path=image_path,
            result=result,
            quality=quality,
            explanation=explanation,
            guidance=guidance,
            device=device,
        )

        print()
        print(
            f"JSON result: {output_path}"
        )

    except Exception as exc:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()