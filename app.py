from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
from PIL import Image, ImageFile
import subprocess
import sys
import shutil
import uuid
import json
import re

from ml.interface import load_model, predict

ImageFile.LOAD_TRUNCATED_IMAGES = False

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent

UPLOAD_DIR = ROOT / "web_uploads"
WEB_OUTPUT_DIR = ROOT / "static" / "generated"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_DEVICE = None
MODEL = None
CLASS_NAMES = None
IMAGE_SIZE = None


def get_model():
    global MODEL_DEVICE
    global MODEL
    global CLASS_NAMES
    global IMAGE_SIZE

    if MODEL is None:
        import torch

        MODEL_DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        MODEL, CLASS_NAMES, IMAGE_SIZE = load_model(
            MODEL_DEVICE
        )

    return MODEL, CLASS_NAMES, IMAGE_SIZE, MODEL_DEVICE


# ------------------------------------------------------------
# IMAGE VALIDATION
# ------------------------------------------------------------

def validate_image(path):
    try:
        if not path.exists():
            return False, "Image file was not found."

        if path.stat().st_size > MAX_FILE_SIZE:
            return False, "Image is too large. Please use an image below 10 MB."

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return False, "Unsupported image format."

        with Image.open(path) as img:
            img.load()

            width, height = img.size

            if width < 64 or height < 64:
                return (
                    False,
                    "This image is too small to classify. "
                    "Try uploading a closer photo of the waste item."
                )

            if width * height < 10000:
                return (
                    False,
                    "This image is too small to classify clearly. "
                    "Try uploading a larger photo."
                )

        return True, "OK"

    except Exception:
        return (
            False,
            "This image could not be read. "
            "Try uploading a normal JPG, PNG or WEBP image."
        )


# ------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------

def run_pipeline(image_path):
    """
    Run the existing ml/pipeline.py.

    We intentionally do not duplicate the pipeline implementation.
    """

    script = ROOT / "ml" / "pipeline.py"

    if not script.exists():
        return {
            "success": False,
            "output": "",
            "error": "ml/pipeline.py was not found."
        }

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(image_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )

        output = result.stdout + "\n" + result.stderr

        return {
            "success": result.returncode == 0,
            "output": output,
            "error": None if result.returncode == 0 else output,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": "The ML pipeline took too long to finish."
        }

    except Exception as exc:
        return {
            "success": False,
            "output": "",
            "error": str(exc),
        }


# ------------------------------------------------------------
# GRAD-CAM
# ------------------------------------------------------------

def run_explanation(image_path):
    """
    Run the existing ml/explain.py.

    The existing script generates the real Grad-CAM image.
    """

    script = ROOT / "ml" / "explain.py"

    if not script.exists():
        return {
            "success": False,
            "path": None,
            "output": "",
            "error": "ml/explain.py was not found."
        }

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(image_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=240,
        )

        output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            return {
                "success": False,
                "path": None,
                "output": output,
                "error": output,
            }

        gradcam_path = find_gradcam(image_path, output)

        if gradcam_path is None:
            return {
                "success": False,
                "path": None,
                "output": output,
                "error": (
                    "The explanation script completed, "
                    "but the Grad-CAM image could not be located."
                ),
            }

        # Copy into a browser-accessible directory.
        output_name = (
            f"{image_path.stem}_gradcam_{uuid.uuid4().hex[:8]}"
            + gradcam_path.suffix
        )

        destination = WEB_OUTPUT_DIR / output_name

        shutil.copy2(
            gradcam_path,
            destination
        )

        return {
            "success": True,
            "path": f"/static/generated/{output_name}",
            "output": output,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "path": None,
            "output": "",
            "error": "Grad-CAM generation took too long."
        }

    except Exception as exc:
        return {
            "success": False,
            "path": None,
            "output": "",
            "error": str(exc),
        }


def find_gradcam(image_path, output):
    """
    Try several ways of finding the actual Grad-CAM output.

    First use the path printed by explain.py.
    Then search the explanations directory.
    """

    # --------------------------------------------------------
    # 1. Look for a path printed by explain.py
    # --------------------------------------------------------

    patterns = [
        r"Grad-CAM explanation\s*:\s*(.+)",
        r"Grad-CAM\s*:\s*(.+)",
        r"explanation\s*:\s*(.+)",
        r"(?:generated|saved)\s+(?:to|at)\s*:\s*(.+)",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            output,
            flags=re.IGNORECASE
        )

        for match in matches:
            candidate = match.strip().strip('"').strip("'")

            candidate = Path(candidate)

            if not candidate.is_absolute():
                candidate = ROOT / candidate

            if candidate.exists() and candidate.is_file():
                return candidate

    # --------------------------------------------------------
    # 2. Search training_outputs/explanations
    # --------------------------------------------------------

    explanation_dir = (
        ROOT
        / "training_outputs"
        / "explanations"
    )

    if not explanation_dir.exists():
        return None

    stem = image_path.stem.lower()

    candidates = []

    for path in explanation_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            continue

        name = path.name.lower()

        score = 0

        if stem in name:
            score += 10

        if "grad" in name:
            score += 5

        if "cam" in name:
            score += 5

        if "why" in name:
            score += 3

        candidates.append(
            (score, path)
        )

    if candidates:
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1].stat().st_mtime
            ),
            reverse=True
        )

        if candidates[0][0] > 0:
            return candidates[0][1]

    return None


# ------------------------------------------------------------
# SIMPLE HUMAN EXPLANATION
# ------------------------------------------------------------

def make_simple_explanation(class_name, confidence):
    """
    This does NOT invent object-specific visual claims.

    It explains what the model output and Grad-CAM actually mean.
    """

    confidence_percent = confidence * 100

    if confidence_percent >= 90:
        confidence_text = (
            "The model is quite confident in this prediction."
        )
    elif confidence_percent >= 70:
        confidence_text = (
            "The model has fairly strong confidence in this prediction."
        )
    else:
        confidence_text = (
            "The model is less certain, so this result should be treated with caution."
        )

    return (
        f"The AI classified this item as {class_name} "
        f"with {confidence_percent:.2f}% confidence. "
        f"{confidence_text} "
        f"It made this decision from visual patterns it learned "
        f"from waste images during training."
    )


# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------
# ANALYZE
# ------------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please select an image."
        }), 400

    uploaded = request.files["image"]

    if not uploaded or not uploaded.filename:
        return jsonify({
            "success": False,
            "error": "Please select an image."
        }), 400

    original_name = Path(uploaded.filename).name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": "Please upload a JPG, PNG or WEBP image."
        }), 400

    unique_name = (
        uuid.uuid4().hex
        + extension
    )

    image_path = UPLOAD_DIR / unique_name

    try:
        uploaded.save(image_path)

        # ----------------------------------------------------
        # IMAGE QUALITY
        # ----------------------------------------------------

        valid, quality_message = validate_image(
            image_path
        )

        if not valid:
            return jsonify({
                "success": False,
                "error": quality_message,
                "quality": "WARNING",
            }), 400

        # ----------------------------------------------------
        # EXISTING PIPELINE
        # ----------------------------------------------------

        pipeline_result = run_pipeline(
            image_path
        )

        if not pipeline_result["success"]:
            return jsonify({
                "success": False,
                "error": (
                    "The ML pipeline could not process this image."
                ),
                "details": pipeline_result["error"],
            }), 500

        # ----------------------------------------------------
        # REAL MODEL PREDICTION
        # ----------------------------------------------------

        model, class_names, image_size, device = get_model()

        predictions = predict(
            model,
            image_path,
            class_names,
            image_size,
            device,
            3,
        )

        if not predictions:
            return jsonify({
                "success": False,
                "error": "The model did not return a prediction."
            }), 500

        best_class, best_confidence = predictions[0]

        top_predictions = []

        for name, confidence in predictions:
            top_predictions.append({
                "class": name,
                "confidence": round(
                    confidence * 100,
                    2
                )
            })

        # ----------------------------------------------------
        # REAL GRAD-CAM
        # ----------------------------------------------------

        explanation_result = run_explanation(
            image_path
        )

        # ----------------------------------------------------
        # COPY ORIGINAL IMAGE FOR BROWSER
        # ----------------------------------------------------

        preview_name = (
            f"{image_path.stem}_preview"
            + image_path.suffix
        )

        preview_path = (
            WEB_OUTPUT_DIR
            / preview_name
        )

        shutil.copy2(
            image_path,
            preview_path
        )

        return jsonify({
            "success": True,

            "image": {
                "name": original_name,
                "url": (
                    "/static/generated/"
                    + preview_name
                ),
            },

            "quality": {
                "status": "OK",
                "message": quality_message,
            },

            "prediction": {
                "class": best_class,
                "confidence": round(
                    best_confidence * 100,
                    2
                ),
            },

            "top_predictions": top_predictions,

            "why_this_result": make_simple_explanation(
                best_class,
                best_confidence,
            ),

            "gradcam": {
                "available": explanation_result["success"],
                "url": explanation_result["path"],
                "message": (
                    None
                    if explanation_result["success"]
                    else explanation_result["error"]
                ),
            },

            "model": "EfficientNet-B0",
            "device": str(device),

        })

    except Exception as exc:

        import traceback

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": (
                "Something went wrong while analyzing the image."
            ),
            "details": str(exc),
        }), 500

    finally:

        # Keep uploaded files for debugging for now.
        # We can add cleanup later.
        pass


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("ENCRYPTOS")
    print("SMART WASTE SEGREGATION AI")
    print("=" * 70)
    print()
    print("Model      : EfficientNet-B0")
    print("Explain    : Grad-CAM")
    print("Pipeline   : ml/pipeline.py")
    print()
    print("Local Demo : http://127.0.0.1:5000")
    print("=" * 70)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )