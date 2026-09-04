from pathlib import Path
import csv
from collections import defaultdict

REPORT_DIR = Path("data/reports")

EXACT_DUPLICATES = REPORT_DIR / "exact_duplicates.csv"
NEAR_DUPLICATES_STRONG = REPORT_DIR / "near_duplicates_strong.csv"
NEAR_DUPLICATES_REVIEW = REPORT_DIR / "near_duplicates_review.csv"


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get_label_from_path(path):
    """
    Attempts to infer the dataset and class label from the file path.
    Expected structure examples:

    trashnet/glass/glass115.jpg
    realwaste/Plastic/image.jpg
    phenomsg/e-waste/image.jpg
    ewaste/train/Printer/image.jpg
    """

    p = Path(path)
    parts = [x.lower() for x in p.parts]

    datasets = ["trashnet", "realwaste", "phenomsg", "ewaste"]

    dataset = None
    dataset_index = None

    for i, part in enumerate(parts):
        if part in datasets:
            dataset = part
            dataset_index = i
            break

    if dataset is None:
        return "", ""

    remaining = parts[dataset_index + 1:]

    if not remaining:
        return dataset, ""

    # E-Waste has train/test/validation before the class
    if dataset == "ewaste" and remaining[0] in {
        "train",
        "test",
        "validation",
        "val",
    }:
        split = remaining[0]
        label = remaining[1] if len(remaining) > 1 else ""
        return dataset, label

    # For the other datasets, first directory after dataset is treated as label
    label = remaining[0]

    return dataset, label


def get_split_from_path(path):
    parts = [x.lower() for x in Path(path).parts]

    for split in ["train", "test", "validation", "val"]:
        if split in parts:
            return split

    return ""


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_exact_duplicates(rows):
    """
    Analyze exact duplicate groups and identify:

    1. Same-label duplicates
    2. Conflicting-label duplicates
    3. Train/test leakage
    4. Cross-dataset duplicates
    """

    groups = defaultdict(list)

    for row in rows:
        # The duplicate detector should have one of these identifiers.
        sha = (
            row.get("sha256")
            or row.get("hash")
            or row.get("file_hash")
            or row.get("duplicate_hash")
        )

        if sha:
            groups[sha].append(row)

    conflicts = []
    leakage = []
    cross_dataset = []

    for sha, group in groups.items():

        enriched = []

        for row in group:
            path = (
                row.get("path")
                or row.get("file_path")
                or row.get("filepath")
                or row.get("file")
                or ""
            )

            dataset, label = get_label_from_path(path)
            split = get_split_from_path(path)

            enriched.append({
                "sha256": sha,
                "dataset": dataset,
                "label": label,
                "split": split,
                "path": path,
            })

        labels = set(
            x["label"]
            for x in enriched
            if x["label"]
        )

        datasets = set(
            x["dataset"]
            for x in enriched
            if x["dataset"]
        )

        splits = set(
            x["split"]
            for x in enriched
            if x["split"]
        )

        # Conflicting labels
        if len(labels) > 1:
            for item in enriched:
                conflicts.append({
                    "sha256": sha,
                    "dataset": item["dataset"],
                    "label": item["label"],
                    "split": item["split"],
                    "path": item["path"],
                    "all_labels": " | ".join(sorted(labels)),
                })

        # Train/test leakage
        if "train" in splits and (
            "test" in splits
            or "validation" in splits
            or "val" in splits
        ):
            for item in enriched:
                leakage.append({
                    "sha256": sha,
                    "dataset": item["dataset"],
                    "label": item["label"],
                    "split": item["split"],
                    "path": item["path"],
                })

        # Cross-dataset duplicates
        if len(datasets) > 1:
            for item in enriched:
                cross_dataset.append({
                    "sha256": sha,
                    "dataset": item["dataset"],
                    "label": item["label"],
                    "split": item["split"],
                    "path": item["path"],
                    "datasets_in_group": " | ".join(sorted(datasets)),
                })

    return conflicts, leakage, cross_dataset


def audit_near_duplicates(rows, filename):
    """
    Creates a normalized report for near duplicates.

    This does NOT delete or modify any images.
    """

    output = REPORT_DIR / filename

    normalized = []

    for row in rows:
        normalized.append({
            "hamming_distance": row.get("hamming_distance")
            or row.get("distance")
            or "",
            "dataset_a": row.get("dataset_a", ""),
            "file_a": row.get("file_a", ""),
            "path_a": row.get("path_a", ""),
            "dataset_b": row.get("dataset_b", ""),
            "file_b": row.get("file_b", ""),
            "path_b": row.get("path_b", ""),
        })

    write_csv(
        output,
        [
            "hamming_distance",
            "dataset_a",
            "file_a",
            "path_a",
            "dataset_b",
            "file_b",
            "path_b",
        ],
        normalized,
    )


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DUPLICATE & LABEL INTEGRITY AUDIT")
    print("=" * 60)

    exact_rows = read_csv(EXACT_DUPLICATES)

    print(f"\nExact duplicate records: {len(exact_rows)}")

    conflicts, leakage, cross_dataset = audit_exact_duplicates(
        exact_rows
    )

    write_csv(
        REPORT_DIR / "duplicate_label_conflicts.csv",
        [
            "sha256",
            "dataset",
            "label",
            "split",
            "path",
            "all_labels",
        ],
        conflicts,
    )

    write_csv(
        REPORT_DIR / "split_leakage.csv",
        [
            "sha256",
            "dataset",
            "label",
            "split",
            "path",
        ],
        leakage,
    )

    write_csv(
        REPORT_DIR / "cross_dataset_duplicates.csv",
        [
            "sha256",
            "dataset",
            "label",
            "split",
            "path",
            "datasets_in_group",
        ],
        cross_dataset,
    )

    # Near duplicate reports
    strong_path = NEAR_DUPLICATES_STRONG

    if strong_path.exists():
        strong_rows = read_csv(strong_path)

        audit_near_duplicates(
            strong_rows,
            "near_duplicates_strong_audit.csv",
        )

        print(
            f"Strong near-duplicate records: {len(strong_rows)}"
        )

    review_path = NEAR_DUPLICATES_REVIEW

    if review_path.exists():
        review_rows = read_csv(review_path)

        audit_near_duplicates(
            review_rows,
            "near_duplicates_review_audit.csv",
        )

        print(
            f"Review near-duplicate records: {len(review_rows)}"
        )

    # Summary
    summary = [
        {
            "metric": "exact_duplicate_records",
            "value": len(exact_rows),
        },
        {
            "metric": "conflicting_label_records",
            "value": len(conflicts),
        },
        {
            "metric": "train_test_leakage_records",
            "value": len(leakage),
        },
        {
            "metric": "cross_dataset_duplicate_records",
            "value": len(cross_dataset),
        },
    ]

    write_csv(
        REPORT_DIR / "duplicate_audit_summary.csv",
        ["metric", "value"],
        summary,
    )

    print("\nAudit complete.")
    print("\nGenerated reports:")

    for filename in [
        "duplicate_label_conflicts.csv",
        "split_leakage.csv",
        "cross_dataset_duplicates.csv",
        "near_duplicates_strong_audit.csv",
        "near_duplicates_review_audit.csv",
        "duplicate_audit_summary.csv",
    ]:
        path = REPORT_DIR / filename
        if path.exists():
            print(f"  ✓ {path}")

    print("\nIMPORTANT:")
    print("No images were deleted or modified.")


if __name__ == "__main__":
    main()