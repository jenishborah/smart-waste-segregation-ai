"""
======================================================================
SMART WASTE SEGREGATION AI
EfficientNet-B0 -> ONNX EXPORT
======================================================================

Exports the trained PyTorch EfficientNet-B0 model to ONNX format
for deployment in Android / ONNX Runtime.

Expected project structure:

smart-waste-segregation-ai/
│
├── ml/
│   └── export_onnx.py
│
├── training_outputs/
│   └── efficientnet_b0_cached/
│       └── best_model.pth
│
├── data/
│   └── ...
│
└── models/
    └── waste_classifier.onnx   <- generated
======================================================================
"""

from pathlib import Path
import json
import sys

import torch
import torch.nn as nn
from torchvision import models


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "training_outputs"
    / "efficientnet_b0_cached"
    / "best_model.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "models"
ONNX_PATH = OUTPUT_DIR / "waste_classifier.onnx"

CLASS_NAMES_PATH = (
    PROJECT_ROOT
    / "training_outputs"
    / "efficientnet_b0_cached"
    / "class_names.json"
)


# ======================================================================
# CLASSES
# ======================================================================

DEFAULT_CLASS_NAMES = [
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


# ======================================================================
# LOAD CLASS NAMES
# ======================================================================

def load_class_names():

    if not CLASS_NAMES_PATH.exists():

        print(
            "class_names.json not found."
        )

        print(
            "Using default class order."
        )

        return DEFAULT_CLASS_NAMES

    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        # Case 1:
        # ["Battery", "Cardboard", ...]
        if isinstance(data, list):

            return data

        # Case 2:
        # {"0": "Battery", "1": "Cardboard", ...}
        if isinstance(data, dict):

            class_names = []

            for i in range(len(data)):

                if str(i) in data:
                    class_names.append(
                        data[str(i)]
                    )

                elif i in data:
                    class_names.append(
                        data[i]
                    )

            if class_names:

                return class_names

    except Exception as e:

        print(
            f"Warning: Could not read class_names.json: {e}"
        )

    print(
        "Using default class order."
    )

    return DEFAULT_CLASS_NAMES


# ======================================================================
# CREATE MODEL
# ======================================================================

def create_model(num_classes):

    print(
        "Creating EfficientNet-B0..."
    )

    model = models.efficientnet_b0(
        weights=None
    )

    # Your trained model uses 16 output classes.
    in_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


# ======================================================================
# LOAD CHECKPOINT
# ======================================================================

def load_checkpoint(model):

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}\n\n"
            "Make sure best_model.pth exists in:\n"
            "training_outputs/efficientnet_b0_cached/"
        )

    print(
        f"Loading model:\n{MODEL_PATH}"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    # --------------------------------------------------------------
    # Different possible checkpoint formats
    # --------------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            # Some training scripts save state_dict directly.
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove "module." prefix if model was trained
    # using DataParallel.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:

        print(
            "\nWARNING: Missing model keys:"
        )

        for key in missing[:20]:

            print(
                f"  {key}"
            )

    if unexpected:

        print(
            "\nWARNING: Unexpected model keys:"
        )

        for key in unexpected[:20]:

            print(
                f"  {key}"
            )

    if missing:

        raise RuntimeError(
            "\nModel weights could not be loaded correctly.\n"
            "The architecture/checkpoint format does not match."
        )

    model.eval()

    return model


# ======================================================================
# VERIFY PYTORCH MODEL
# ======================================================================

def verify_pytorch_model(
    model,
    class_names
):

    print(
        "\n" + "=" * 70
    )

    print(
        "VERIFYING PYTORCH MODEL"
    )

    print(
        "=" * 70
    )

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print(
        f"Input shape : {tuple(dummy_input.shape)}"
    )

    print(
        f"Output shape: {tuple(output.shape)}"
    )

    expected_shape = (
        1,
        len(class_names)
    )

    if tuple(output.shape) != expected_shape:

        raise RuntimeError(
            f"\nUnexpected output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got     : {tuple(output.shape)}"
        )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_index = (
        probabilities.argmax(dim=1).item()
    )

    confidence = (
        probabilities[0, predicted_index].item()
    )

    print(
        f"Test prediction : {class_names[predicted_index]}"
    )

    print(
        f"Test confidence : {confidence * 100:.2f}%"
    )

    print(
        "PyTorch verification: SUCCESS"
    )


# ======================================================================
# EXPORT ONNX
# ======================================================================

def export_model(
    model,
    class_names
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPORTING TO ONNX"
    )

    print(
        "=" * 70
    )

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    print(
        f"Output:\n{ONNX_PATH}"
    )

    # --------------------------------------------------------------
    # ONNX export
    # --------------------------------------------------------------

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),

        export_params=True,

        opset_version=17,

        do_constant_folding=True,

        input_names=[
            "input"
        ],

        output_names=[
            "output"
        ],

        dynamic_axes={
            "input": {
                0: "batch_size"
            },
            "output": {
                0: "batch_size"
            },
        },
    )

    print(
        "\nONNX export completed."
    )

    # --------------------------------------------------------------
    # Save class labels beside ONNX model
    # --------------------------------------------------------------

    labels_path = (
        OUTPUT_DIR
        / "class_names.json"
    )

    with open(
        labels_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            class_names,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Class names:\n{labels_path}"
    )


# ======================================================================
# VERIFY ONNX
# ======================================================================

def verify_onnx(
    model,
    class_names
):

    print(
        "\n" + "=" * 70
    )

    print(
        "VERIFYING ONNX MODEL"
    )

    print(
        "=" * 70
    )

    try:

        import onnx
        import onnxruntime as ort
        import numpy as np

    except ImportError:

        print(
            "\nONNX verification packages are not installed."
        )

        print(
            "Install them with:"
        )

        print(
            "pip install onnx onnxruntime"
        )

        print(
            "\nSkipping ONNX runtime verification."
        )

        return

    # --------------------------------------------------------------
    # Validate ONNX structure
    # --------------------------------------------------------------

    onnx_model = onnx.load(
        str(ONNX_PATH)
    )

    onnx.checker.check_model(
        onnx_model
    )

    print(
        "ONNX structural check: SUCCESS"
    )

    # --------------------------------------------------------------
    # Create ONNX Runtime session
    # --------------------------------------------------------------

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CPUExecutionProvider"
        ]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(
        f"ONNX input : {input_name}"
    )

    print(
        f"ONNX output: {output_name}"
    )

    # --------------------------------------------------------------
    # Same deterministic input for both models
    # --------------------------------------------------------------

    torch.manual_seed(1234)

    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    # PyTorch prediction
    with torch.no_grad():

        torch_output = model(
            dummy_input
        )

        torch_probabilities = torch.softmax(
            torch_output,
            dim=1
        ).numpy()

    # ONNX prediction
    numpy_input = (
        dummy_input.numpy()
        .astype(np.float32)
    )

    onnx_output = session.run(
        [output_name],
        {
            input_name: numpy_input
        }
    )[0]

    # ONNX output is logits
    onnx_exp = np.exp(
        onnx_output
        - np.max(
            onnx_output,
            axis=1,
            keepdims=True
        )
    )

    onnx_probabilities = (
        onnx_exp
        / np.sum(
            onnx_exp,
            axis=1,
            keepdims=True
        )
    )

    # --------------------------------------------------------------
    # Compare predictions
    # --------------------------------------------------------------

    torch_index = int(
        np.argmax(
            torch_probabilities,
            axis=1
        )[0]
    )

    onnx_index = int(
        np.argmax(
            onnx_probabilities,
            axis=1
        )[0]
    )

    torch_confidence = float(
        torch_probabilities[
            0,
            torch_index
        ]
    )

    onnx_confidence = float(
        onnx_probabilities[
            0,
            onnx_index
        ]
    )

    max_difference = float(
        np.max(
            np.abs(
                torch_probabilities
                - onnx_probabilities
            )
        )
    )

    print(
        "\nPYTORCH"
    )

    print(
        f"Class      : {class_names[torch_index]}"
    )

    print(
        f"Confidence : {torch_confidence * 100:.2f}%"
    )

    print(
        "\nONNX"
    )

    print(
        f"Class      : {class_names[onnx_index]}"
    )

    print(
        f"Confidence : {onnx_confidence * 100:.2f}%"
    )

    print(
        f"\nMaximum probability difference: "
        f"{max_difference:.8f}"
    )

    if torch_index == onnx_index:

        print(
            "\nPrediction consistency: SUCCESS"
        )

    else:

        raise RuntimeError(
            "\nERROR: PyTorch and ONNX predictions differ."
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "SMART WASTE SEGREGATION AI"
    )

    print(
        "EFFICIENTNET-B0 -> ONNX EXPORT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nProject root:"
        f"\n{PROJECT_ROOT}"
    )

    print(
        f"\nPyTorch model:"
        f"\n{MODEL_PATH}"
    )

    print(
        f"\nONNX output:"
        f"\n{ONNX_PATH}"
    )

    # --------------------------------------------------------------
    # Load classes
    # --------------------------------------------------------------

    class_names = load_class_names()

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASSES"
    )

    print(
        "=" * 70
    )

    for index, name in enumerate(
        class_names
    ):

        print(
            f"{index:2d}  {name}"
        )

    print(
        f"\nNumber of classes: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------

    model = create_model(
        len(class_names)
    )

    print(
        f"\nParameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------------

    model = load_checkpoint(
        model
    )

    print(
        "\nTrained weights loaded successfully."
    )

    # --------------------------------------------------------------
    # Verify PyTorch
    # --------------------------------------------------------------

    verify_pytorch_model(
        model,
        class_names
    )

    # --------------------------------------------------------------
    # Export
    # --------------------------------------------------------------

    export_model(
        model,
        class_names
    )

    # --------------------------------------------------------------
    # Verify ONNX
    # --------------------------------------------------------------

    verify_onnx(
        model,
        class_names
    )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPORT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nONNX model:"
        f"\n{ONNX_PATH}"
    )

    print(
        f"\nClass labels:"
        f"\n{OUTPUT_DIR / 'class_names.json'}"
    )

    print(
        "\nYou can now use the ONNX model for"
        " Android inference."
    )

    print(
        "\nSTATUS: SUCCESS"
    )


if __name__ == "__main__":
    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nExport cancelled."
        )

        sys.exit(1)

    except Exception as e:

        print(
            "\n" + "=" * 70
        )

        print(
            "ERROR"
        )

        print(
            "=" * 70
        )

        print(
            f"\n{e}"
        )

        sys.exit(1)