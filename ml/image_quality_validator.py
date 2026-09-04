from pathlib import Path
from PIL import Image, ImageFile
import csv
import statistics

# Allow PIL to report truncated images instead of silently failing
ImageFile.LOAD_TRUNCATED_IMAGES = False

DATA_ROOT = Path("data/raw")
REPORT_DIR = Path("data/reports")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# Thresholds are intentionally conservative.
MIN_WIDTH = 64
MIN_HEIGHT = 64

EXTREME_ASPECT_RATIO = 5.0

# Image-quality warning thresholds
BLUR_THRESHOLD = 50.0
VERY_DARK_THRESHOLD = 25.0
VERY_BRIGHT_THRESHOLD = 235.0


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_dataset(path):
    parts = [p.lower() for p in path.parts]

    for dataset in ["trashnet", "realwaste", "phenomsg", "ewaste"]:
        if dataset in parts:
            return dataset

    return "unknown"


def get_label(path):
    parts = list(path.parts)
    lower_parts = [p.lower() for p in parts]

    dataset = get_dataset(path)

    try:
        index = lower_parts.index(dataset)
    except ValueError:
        return ""

    remaining = parts[index + 1:]

    if not remaining:
        return ""

    # E-Waste structure:
    # ewaste/train/Printer/image.jpg
    # ewaste/test/Printer/image.jpg
    # ewaste/validation/Printer/image.jpg
    if dataset == "ewaste":
        if remaining[0].lower() in {
            "train",
            "test",
            "validation",
            "val",
        }:
            return remaining[1] if len(remaining) > 1 else ""

    return remaining[0]


def get_split(path):
    parts = [p.lower() for p in path.parts]

    for split in ["train", "test", "validation", "val"]:
        if split in parts:
            return split

    return ""


def calculate_blur_score(image):
    """
    Uses variance of the Laplacian as a simple sharpness/blur indicator.
    Higher generally means sharper.
    """

    try:
        import cv2

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    except ImportError:
        return None

    except Exception:
        return None


def analyze_image(path):
    result = {
        "dataset": get_dataset(path),
        "label": get_label(path),
        "split": get_split(path),
        "path": str(path),

        "width": "",
        "height": "",
        "aspect_ratio": "",
        "format": "",
        "mode": "",

        "mean_brightness": "",
        "blur_score": "",

        "status": "OK",
        "issues": "",
    }

    issues = []

    try:
        with Image.open(path) as img:

            result["format"] = img.format or ""
            result["mode"] = img.mode

            # Force complete decoding.
            # This catches truncated/corrupt files.
            img.load()

            rgb = img.convert("RGB")

            width, height = rgb.size

            result["width"] = width
            result["height"] = height

            if height > 0:
                aspect_ratio = width / height
            else:
                aspect_ratio = 0

            result["aspect_ratio"] = round(aspect_ratio, 4)

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                issues.append("very_small")

            if (
                aspect_ratio > EXTREME_ASPECT_RATIO
                or aspect_ratio < 1 / EXTREME_ASPECT_RATIO
            ):
                issues.append("extreme_aspect_ratio")

            # Brightness
            gray = rgb.convert("L")

            pixels = list(gray.getdata())

            if pixels:
                mean_brightness = statistics.mean(pixels)
                result["mean_brightness"] = round(
                    mean_brightness,
                    2
                )

                if mean_brightness < VERY_DARK_THRESHOLD:
                    issues.append("very_dark")

                elif mean_brightness > VERY_BRIGHT_THRESHOLD:
                    issues.append("very_bright")

            # Blur
            blur_score = calculate_blur_score(
                rgb
            )

            if blur_score is not None:

                result["blur_score"] = round(
                    blur_score,
                    2
                )

                if blur_score < BLUR_THRESHOLD:
                    issues.append("possible_blur")

            # Transparency
            if "transparency" in img.info:
                issues.append("transparency")

            if img.mode in {
                "RGBA",
                "LA",
                "P",
            }:
                if "transparency" not in issues:
                    issues.append("non_rgb_mode")

    except Exception as e:

        result["status"] = "ERROR"
        issues.append(
            f"read_error:{type(e).__name__}"
        )

    if issues and result["status"] == "OK":
        result["status"] = "WARNING"

    result["issues"] = " | ".join(issues)

    return result


def main():

    print("=" * 70)
    print("IMAGE QUALITY VALIDATION")
    print("=" * 70)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = []

    for path in DATA_ROOT.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            image_paths.append(path)

    image_paths.sort()

    print(
        f"\nImages found: {len(image_paths)}"
    )

    results = []

    for index, path in enumerate(
        image_paths,
        start=1
    ):

        result = analyze_image(path)
        results.append(result)

        if index % 500 == 0:
            print(
                f"Processed {index}/{len(image_paths)}"
            )

    # Full report
    all_fields = [
        "dataset",
        "label",
        "split",
        "path",
        "width",
        "height",
        "aspect_ratio",
        "format",
        "mode",
        "mean_brightness",
        "blur_score",
        "status",
        "issues",
    ]

    write_csv(
        REPORT_DIR / "image_quality_report.csv",
        all_fields,
        results,
    )

    # Problematic images
    problematic = [
        row
        for row in results
        if row["status"] != "OK"
    ]

    write_csv(
        REPORT_DIR / "image_quality_issues.csv",
        all_fields,
        problematic,
    )

    # Corrupt/unreadable
    corrupt = [
        row
        for row in results
        if row["status"] == "ERROR"
    ]

    write_csv(
        REPORT_DIR / "image_quality_corrupt.csv",
        all_fields,
        corrupt,
    )

    # Summary
    summary = []

    total = len(results)

    ok = sum(
        1
        for r in results
        if r["status"] == "OK"
    )

    warnings = sum(
        1
        for r in results
        if r["status"] == "WARNING"
    )

    errors = sum(
        1
        for r in results
        if r["status"] == "ERROR"
    )

    summary.extend([
        {
            "metric": "total_images",
            "value": total,
        },
        {
            "metric": "ok_images",
            "value": ok,
        },
        {
            "metric": "warning_images",
            "value": warnings,
        },
        {
            "metric": "error_images",
            "value": errors,
        },
    ])

    # Issue counts
    issue_counts = {}

    for row in results:

        if row["issues"]:

            for issue in row["issues"].split(" | "):

                if issue.startswith("read_error:"):
                    key = "read_error"
                else:
                    key = issue

                issue_counts[key] = (
                    issue_counts.get(key, 0)
                    + 1
                )

    for issue, count in sorted(
        issue_counts.items()
    ):

        summary.append({
            "metric": f"issue_{issue}",
            "value": count,
        })

    write_csv(
        REPORT_DIR / "image_quality_summary.csv",
        ["metric", "value"],
        summary,
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal images: {total}"
    )

    print(
        f"OK: {ok}"
    )

    print(
        f"Warnings: {warnings}"
    )

    print(
        f"Errors: {errors}"
    )

    print("\nReports generated:")

    for filename in [
        "image_quality_summary.csv",
        "image_quality_report.csv",
        "image_quality_issues.csv",
        "image_quality_corrupt.csv",
    ]:

        print(
            f"  ✓ data/reports/{filename}"
        )

    print(
        "\nNo images were modified or deleted."
    )


if __name__ == "__main__":
    main()