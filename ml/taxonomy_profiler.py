from pathlib import Path
from collections import Counter, defaultdict
import csv

# ============================================================
# CONFIGURATION
# ============================================================

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

DATASETS = [
    "trashnet",
    "realwaste",
    "phenomsg",
    "ewaste",
]


# ============================================================
# HELPERS
# ============================================================

def get_dataset(path: Path) -> str:
    """
    Identify which dataset an image belongs to.
    """

    parts = [p.lower() for p in path.parts]

    for dataset in DATASETS:
        if dataset in parts:
            return dataset

    return "unknown"


def get_class(path: Path) -> str:
    """
    Extract the original dataset class/label.

    Handles common structures such as:

        trashnet/<class>/image.jpg

        realwaste/<class>/image.jpg

        phenomsg/<class>/image.jpg

        ewaste/train/<class>/image.jpg
        ewaste/test/<class>/image.jpg
        ewaste/validation/<class>/image.jpg
    """

    dataset = get_dataset(path)

    parts = list(path.parts)
    lower_parts = [p.lower() for p in parts]

    try:
        dataset_index = lower_parts.index(dataset)
    except ValueError:
        return "UNKNOWN"

    remaining = parts[dataset_index + 1:]

    if not remaining:
        return "UNKNOWN"

    # E-Waste commonly contains split directories
    # before the actual class.
    if dataset == "ewaste":

        if remaining[0].lower() in {
            "train",
            "test",
            "validation",
            "val",
        }:

            if len(remaining) >= 2:
                return remaining[1]

            return "UNKNOWN"

    # For the other datasets, first directory
    # after dataset is normally the class.
    return remaining[0]


def get_split(path: Path) -> str:
    """
    Detect train/validation/test if present.
    """

    parts = [p.lower() for p in path.parts]

    if "train" in parts:
        return "train"

    if "validation" in parts:
        return "validation"

    if "val" in parts:
        return "validation"

    if "test" in parts:
        return "test"

    return "unspecified"


def write_csv(
    filepath: Path,
    fieldnames,
    rows,
):
    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        filepath,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# MAIN PROFILING
# ============================================================

def main():

    print("=" * 75)
    print("WASTE SEGREGATION AI")
    print("UNIFIED TAXONOMY SOURCE PROFILER")
    print("=" * 75)

    print(f"\nData root:")
    print(DATA_ROOT.resolve())

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

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
        f"\nTotal images discovered: {len(image_paths):,}"
    )

    if not image_paths:
        print("\nERROR: No images found.")
        print("Check your data/raw directory.")
        return

    # --------------------------------------------------------
    # Collect records
    # --------------------------------------------------------

    records = []

    dataset_counter = Counter()
    dataset_class_counter = Counter()
    dataset_split_counter = Counter()

    for path in image_paths:

        dataset = get_dataset(path)
        original_class = get_class(path)
        split = get_split(path)

        dataset_counter[dataset] += 1

        dataset_class_counter[
            (dataset, original_class)
        ] += 1

        dataset_split_counter[
            (dataset, split)
        ] += 1

        records.append({
            "dataset": dataset,
            "original_class": original_class,
            "split": split,
            "image_count": 1,
            "relative_path": str(
                path.relative_to(DATA_ROOT)
            ),
        })

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    dataset_rows = []

    total_images = len(image_paths)

    for dataset, count in sorted(
        dataset_counter.items()
    ):

        percentage = (
            count / total_images * 100
        )

        dataset_rows.append({
            "dataset": dataset,
            "image_count": count,
            "percentage": round(
                percentage,
                2,
            ),
        })

    write_csv(
        REPORT_DIR / "taxonomy_dataset_counts.csv",
        [
            "dataset",
            "image_count",
            "percentage",
        ],
        dataset_rows,
    )

    # --------------------------------------------------------
    # Dataset + class summary
    # --------------------------------------------------------

    class_rows = []

    for (
        dataset,
        original_class,
    ), count in sorted(
        dataset_class_counter.items()
    ):

        dataset_total = dataset_counter[
            dataset
        ]

        percentage = (
            count / dataset_total * 100
        )

        class_rows.append({
            "dataset": dataset,
            "original_class": original_class,
            "image_count": count,
            "percentage_within_dataset": round(
                percentage,
                2,
            ),
        })

    write_csv(
        REPORT_DIR / "taxonomy_source_counts.csv",
        [
            "dataset",
            "original_class",
            "image_count",
            "percentage_within_dataset",
        ],
        class_rows,
    )

    # --------------------------------------------------------
    # Dataset + class + split summary
    # --------------------------------------------------------

    split_rows = []

    split_class_counter = Counter()

    for record in records:

        key = (
            record["dataset"],
            record["original_class"],
            record["split"],
        )

        split_class_counter[key] += 1

    for (
        dataset,
        original_class,
        split,
    ), count in sorted(
        split_class_counter.items()
    ):

        split_rows.append({
            "dataset": dataset,
            "original_class": original_class,
            "split": split,
            "image_count": count,
        })

    write_csv(
        REPORT_DIR / "taxonomy_split_counts.csv",
        [
            "dataset",
            "original_class",
            "split",
            "image_count",
        ],
        split_rows,
    )

    # --------------------------------------------------------
    # Raw image inventory
    # --------------------------------------------------------

    write_csv(
        REPORT_DIR / "taxonomy_image_inventory.csv",
        [
            "dataset",
            "original_class",
            "split",
            "image_count",
            "relative_path",
        ],
        records,
    )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print("\n" + "=" * 75)
    print("DATASET SUMMARY")
    print("=" * 75)

    for row in dataset_rows:

        print(
            f"{row['dataset']:12s}"
            f" {row['image_count']:6,}"
            f" ({row['percentage']:6.2f}%)"
        )

    print("\n" + "=" * 75)
    print("ORIGINAL CLASS DISTRIBUTION")
    print("=" * 75)

    current_dataset = None

    for row in class_rows:

        dataset = row["dataset"]

        if dataset != current_dataset:

            print(
                f"\n[{dataset.upper()}]"
            )

            current_dataset = dataset

        print(
            f"  {row['original_class']:<30}"
            f"{row['image_count']:6,}"
            f" ({row['percentage_within_dataset']:6.2f}%)"
        )

    print("\n" + "=" * 75)
    print("SPLIT DISTRIBUTION")
    print("=" * 75)

    current_dataset = None
    current_class = None

    for row in split_rows:

        key = (
            row["dataset"],
            row["original_class"],
        )

        if key != (
            current_dataset,
            current_class,
        ):

            print(
                f"\n[{row['dataset'].upper()}]"
            )

            print(
                f"  {row['original_class']}"
            )

            current_dataset = row["dataset"]
            current_class = row[
                "original_class"
            ]

        print(
            f"    {row['split']:<15}"
            f"{row['image_count']:6,}"
        )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("REPORTS CREATED")
    print("=" * 75)

    reports = [
        "taxonomy_dataset_counts.csv",
        "taxonomy_source_counts.csv",
        "taxonomy_split_counts.csv",
        "taxonomy_image_inventory.csv",
    ]

    for report in reports:

        print(
            f"  ✓ data/reports/{report}"
        )

    print("\nNo dataset files were modified.")
    print("No files were moved.")
    print("No files were deleted.")

    print(
        "\nTaxonomy profiling complete."
    )


if __name__ == "__main__":
    main()