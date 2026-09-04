"""
Waste Segregation AI
====================

Dataset Profiler

Purpose:
    Analyze the downloaded waste datasets locally before
    preprocessing and model training.

Datasets:
    - TrashNet
    - RealWaste
    - PhenomSG
    - E-Waste

IMPORTANT:
    This script only READS the datasets.

    It does NOT:
        - modify images
        - rename images
        - move images
        - delete images
        - resize images
        - create train/validation/test splits
"""

from pathlib import Path
from collections import Counter
from statistics import mean, median
import csv

from PIL import Image


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "trashnet": RAW_DATA_DIR / "trashnet",
    "realwaste": RAW_DATA_DIR / "realwaste",
    "phenomsg": RAW_DATA_DIR / "phenomsg",
    "ewaste": RAW_DATA_DIR / "ewaste",
}


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

MIN_WIDTH = 100
MIN_HEIGHT = 100


# ============================================================
# SPLIT FOLDERS
# ============================================================

SPLIT_FOLDERS = {
    "train",
    "training",
    "test",
    "testing",
    "val",
    "valid",
    "validation",
}


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(dataset_path):
    """
    Recursively find all supported image files.
    """

    images = []

    for path in dataset_path.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


# ============================================================
# DETECT CLASS
# ============================================================

def detect_class(image_path, dataset_path):
    """
    Detect the class name from the directory structure.

    Examples:

        trashnet/glass/image.jpg
            -> glass

        realwaste/Plastic/image.jpg
            -> Plastic

        ewaste/train/Mobile/image.jpg
            -> Mobile

        ewaste/validation/Keyboard/image.jpg
            -> Keyboard

        phenomsg/Hazardous/Batteries/image.jpg
            -> Batteries
    """

    relative_path = image_path.relative_to(dataset_path)

    parts = relative_path.parts

    if len(parts) < 2:
        return "UNKNOWN"

    # Remove filename
    directories = list(parts[:-1])

    # Walk backwards through folders
    for folder in reversed(directories):

        if folder.lower() not in SPLIT_FOLDERS:
            return folder

    return "UNKNOWN"


# ============================================================
# DETECT SPLIT
# ============================================================

def detect_split(image_path, dataset_path):
    """
    Detect whether an image belongs to train/test/validation.

    Returns:
        train
        validation
        test
        unspecified
    """

    relative_path = image_path.relative_to(dataset_path)

    parts = relative_path.parts

    for part in parts:

        normalized = part.lower()

        if normalized in {
            "train",
            "training",
        }:
            return "train"

        if normalized in {
            "val",
            "valid",
            "validation",
        }:
            return "validation"

        if normalized in {
            "test",
            "testing",
        }:
            return "test"

    return "unspecified"


# ============================================================
# INSPECT IMAGE
# ============================================================

def inspect_image(image_path):
    """
    Validate an image and collect basic metadata.
    """

    try:

        # First verify the image
        with Image.open(image_path) as image:
            image.verify()

        # Re-open after verify
        with Image.open(image_path) as image:

            width, height = image.size
            image_format = image.format or ""
            mode = image.mode or ""

        aspect_ratio = (
            width / height
            if height > 0
            else 0
        )

        return {
            "valid": True,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "format": image_format,
            "mode": mode,
            "error": "",
        }

    except Exception as error:

        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "aspect_ratio": 0,
            "format": "",
            "mode": "",
            "error": str(error),
        }


# ============================================================
# PROFILE ONE DATASET
# ============================================================

def profile_dataset(dataset_name, dataset_path):

    print("\n")
    print("=" * 75)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 75)

    if not dataset_path.exists():

        print(
            f"[WARNING] Dataset folder not found:"
        )

        print(
            f"          {dataset_path}"
        )

        return None

    images = find_images(dataset_path)

    print(
        f"Images found: {len(images):,}"
    )

    if not images:

        print(
            "[WARNING] No supported image files found."
        )

        return None

    # Counters
    class_counts = Counter()
    split_counts = Counter()
    extension_counts = Counter()
    format_counts = Counter()
    mode_counts = Counter()
    dimension_counts = Counter()

    widths = []
    heights = []
    aspect_ratios = []

    corrupt_files = []
    small_files = []

    valid_images = 0

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for index, image_path in enumerate(
        images,
        start=1
    ):

        # Detect class
        class_name = detect_class(
            image_path,
            dataset_path
        )

        class_counts[class_name] += 1

        # Detect train/test/validation
        split = detect_split(
            image_path,
            dataset_path
        )

        split_counts[split] += 1

        # File extension
        extension = image_path.suffix.lower()

        extension_counts[extension] += 1

        # Image metadata
        info = inspect_image(image_path)

        # ----------------------------------------------------
        # Corrupt image
        # ----------------------------------------------------

        if not info["valid"]:

            corrupt_files.append({
                "dataset": dataset_name,
                "class_name": class_name,
                "split": split,
                "file": str(
                    image_path.relative_to(dataset_path)
                ),
                "error": info["error"],
            })

            continue

        valid_images += 1

        width = info["width"]
        height = info["height"]

        widths.append(width)
        heights.append(height)
        aspect_ratios.append(
            info["aspect_ratio"]
        )

        format_counts[
            info["format"]
        ] += 1

        mode_counts[
            info["mode"]
        ] += 1

        dimension_counts[
            f"{width}x{height}"
        ] += 1

        # ----------------------------------------------------
        # Small image
        # ----------------------------------------------------

        if (
            width < MIN_WIDTH
            or height < MIN_HEIGHT
        ):

            small_files.append({
                "dataset": dataset_name,
                "class_name": class_name,
                "split": split,
                "file": str(
                    image_path.relative_to(dataset_path)
                ),
                "width": width,
                "height": height,
            })

        # Progress
        if index % 500 == 0:

            print(
                f"  Processed "
                f"{index:,}/{len(images):,}"
            )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print("\nCLASS DISTRIBUTION")
    print("-" * 75)

    for class_name, count in sorted(
        class_counts.items(),
        key=lambda x: x[0].lower()
    ):

        print(
            f"{class_name:<35}"
            f"{count:>8,}"
        )

    print("\nDATA SPLITS")
    print("-" * 75)

    for split, count in sorted(
        split_counts.items()
    ):

        print(
            f"{split:<20}"
            f"{count:>8,}"
        )

    print("\nFILE EXTENSIONS")
    print("-" * 75)

    for extension, count in sorted(
        extension_counts.items()
    ):

        print(
            f"{extension:<15}"
            f"{count:>8,}"
        )

    print("\nIMAGE QUALITY")
    print("-" * 75)

    print(
        f"Valid images:       {valid_images:,}"
    )

    print(
        f"Corrupt images:     {len(corrupt_files):,}"
    )

    print(
        f"Very small images:  {len(small_files):,}"
    )

    # --------------------------------------------------------
    # Dimension statistics
    # --------------------------------------------------------

    if widths:

        print("\nIMAGE DIMENSIONS")
        print("-" * 75)

        print(
            f"Width:"
            f" min={min(widths)},"
            f" max={max(widths)},"
            f" mean={mean(widths):.2f},"
            f" median={median(widths)}"
        )

        print(
            f"Height:"
            f" min={min(heights)},"
            f" max={max(heights)},"
            f" mean={mean(heights):.2f},"
            f" median={median(heights)}"
        )

        print(
            f"Aspect ratio:"
            f" min={min(aspect_ratios):.3f},"
            f" max={max(aspect_ratios):.3f},"
            f" mean={mean(aspect_ratios):.3f}"
        )

    # --------------------------------------------------------
    # Common dimensions
    # --------------------------------------------------------

    print("\nMOST COMMON DIMENSIONS")
    print("-" * 75)

    for dimension, count in (
        dimension_counts.most_common(10)
    ):

        print(
            f"{dimension:<20}"
            f"{count:>8,}"
        )

    return {
        "dataset": dataset_name,
        "path": str(dataset_path),
        "total_images": len(images),
        "valid_images": valid_images,
        "corrupt_images": len(corrupt_files),
        "small_images": len(small_files),
        "classes": dict(class_counts),
        "splits": dict(split_counts),
        "extensions": dict(extension_counts),
        "formats": dict(format_counts),
        "modes": dict(mode_counts),
        "dimensions": dict(dimension_counts),
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
        "corrupt_files": corrupt_files,
        "small_files": small_files,
    }


# ============================================================
# CSV WRITER
# ============================================================

def write_csv(
    file_path,
    headers,
    rows
):
    """
    Write rows to CSV.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=headers
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# CREATE REPORTS
# ============================================================

def create_reports(results):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    summary_rows = []

    for result in results:

        summary_rows.append({
            "dataset": result["dataset"],
            "total_images": result["total_images"],
            "valid_images": result["valid_images"],
            "corrupt_images": result["corrupt_images"],
            "small_images": result["small_images"],
            "number_of_classes": len(
                result["classes"]
            ),
        })

    write_csv(
        REPORT_DIR / "dataset_summary.csv",
        [
            "dataset",
            "total_images",
            "valid_images",
            "corrupt_images",
            "small_images",
            "number_of_classes",
        ],
        summary_rows
    )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    class_rows = []

    for result in results:

        for class_name, count in sorted(
            result["classes"].items(),
            key=lambda x: x[0].lower()
        ):

            class_rows.append({
                "dataset": result["dataset"],
                "class_name": class_name,
                "image_count": count,
            })

    write_csv(
        REPORT_DIR / "class_distribution.csv",
        [
            "dataset",
            "class_name",
            "image_count",
        ],
        class_rows
    )

    # ========================================================
    # SPLIT DISTRIBUTION
    # ========================================================

    split_rows = []

    for result in results:

        for split, count in sorted(
            result["splits"].items()
        ):

            split_rows.append({
                "dataset": result["dataset"],
                "split": split,
                "image_count": count,
            })

    write_csv(
        REPORT_DIR / "split_distribution.csv",
        [
            "dataset",
            "split",
            "image_count",
        ],
        split_rows
    )

    # ========================================================
    # IMAGE STATISTICS
    # ========================================================

    statistics_rows = []

    for result in results:

        widths = result["widths"]
        heights = result["heights"]
        ratios = result["aspect_ratios"]

        if not widths:
            continue

        statistics_rows.append({
            "dataset": result["dataset"],
            "min_width": min(widths),
            "max_width": max(widths),
            "mean_width": round(
                mean(widths),
                2
            ),
            "median_width": median(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "mean_height": round(
                mean(heights),
                2
            ),
            "median_height": median(heights),
            "min_aspect_ratio": round(
                min(ratios),
                4
            ),
            "max_aspect_ratio": round(
                max(ratios),
                4
            ),
            "mean_aspect_ratio": round(
                mean(ratios),
                4
            ),
        })

    write_csv(
        REPORT_DIR / "image_statistics.csv",
        [
            "dataset",
            "min_width",
            "max_width",
            "mean_width",
            "median_width",
            "min_height",
            "max_height",
            "mean_height",
            "median_height",
            "min_aspect_ratio",
            "max_aspect_ratio",
            "mean_aspect_ratio",
        ],
        statistics_rows
    )

    # ========================================================
    # CORRUPT IMAGES
    # ========================================================

    corrupt_rows = []

    for result in results:

        corrupt_rows.extend(
            result["corrupt_files"]
        )

    write_csv(
        REPORT_DIR / "corrupt_images.csv",
        [
            "dataset",
            "class_name",
            "split",
            "file",
            "error",
        ],
        corrupt_rows
    )

    # ========================================================
    # SMALL IMAGES
    # ========================================================

    small_rows = []

    for result in results:

        small_rows.extend(
            result["small_files"]
        )

    write_csv(
        REPORT_DIR / "small_images.csv",
        [
            "dataset",
            "class_name",
            "split",
            "file",
            "width",
            "height",
        ],
        small_rows
    )

    # ========================================================
    # DIMENSION DISTRIBUTION
    # ========================================================

    dimension_rows = []

    for result in results:

        for dimension, count in sorted(
            result["dimensions"].items()
        ):

            dimension_rows.append({
                "dataset": result["dataset"],
                "dimension": dimension,
                "image_count": count,
            })

    write_csv(
        REPORT_DIR / "dimension_distribution.csv",
        [
            "dataset",
            "dimension",
            "image_count",
        ],
        dimension_rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("WASTE SEGREGATION AI")
    print("DATASET PROFILER")
    print("=" * 75)

    print(
        "\nProject root:"
    )

    print(PROJECT_ROOT)

    print(
        "\nRaw dataset directory:"
    )

    print(RAW_DATA_DIR)

    print("\nChecking datasets...")

    results = []

    for dataset_name, dataset_path in DATASETS.items():

        print(
            f"  {dataset_name:<15}",
            end=""
        )

        if dataset_path.exists():

            print("[FOUND]")

        else:

            print("[NOT FOUND]")

    # ========================================================
    # PROFILE
    # ========================================================

    for dataset_name, dataset_path in DATASETS.items():

        result = profile_dataset(
            dataset_name,
            dataset_path
        )

        if result is not None:

            results.append(result)

    if not results:

        print(
            "\nERROR: No datasets could be profiled."
        )

        return

    # ========================================================
    # CREATE REPORTS
    # ========================================================

    create_reports(results)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_images = sum(
        result["total_images"]
        for result in results
    )

    total_valid = sum(
        result["valid_images"]
        for result in results
    )

    total_corrupt = sum(
        result["corrupt_images"]
        for result in results
    )

    print("\n")
    print("=" * 75)
    print("PROFILING COMPLETE")
    print("=" * 75)

    print(
        f"\nDatasets analyzed: {len(results)}"
    )

    print(
        f"Total images:      {total_images:,}"
    )

    print(
        f"Valid images:      {total_valid:,}"
    )

    print(
        f"Corrupt images:    {total_corrupt:,}"
    )

    print(
        "\nReports saved to:"
    )

    print(REPORT_DIR)

    print("\nGenerated reports:")

    print(
        "  1. dataset_summary.csv"
    )

    print(
        "  2. class_distribution.csv"
    )

    print(
        "  3. split_distribution.csv"
    )

    print(
        "  4. image_statistics.csv"
    )

    print(
        "  5. corrupt_images.csv"
    )

    print(
        "  6. small_images.csv"
    )

    print(
        "  7. dimension_distribution.csv"
    )

    print(
        "\nOriginal datasets were NOT modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()