"""
WASTE SEGREGATION AI
FINAL DATASET V1 AUDIT

Purpose:
    Perform a final integrity and quality audit of the curated
    waste-segregation dataset before model training.

Important:
    - Does NOT modify data/raw
    - Does NOT delete images
    - Does NOT move images
    - Does NOT rename images
    - Only reads datasets and creates audit reports
"""

from pathlib import Path
import json
import math
import sys

import pandas as pd


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORT_DIR = DATA_DIR / "reports"

MANIFEST_PATH = REPORT_DIR / "dataset_v1_manifest.csv"
CURATED_MANIFEST_PATH = REPORT_DIR / "dataset_v1_curated_manifest.csv"

EXACT_DUPLICATES_PATH = REPORT_DIR / "exact_duplicates.csv"
NEAR_DUPLICATES_PATH = REPORT_DIR / "near_duplicates_strong.csv"
CONFLICTS_PATH = REPORT_DIR / "label_conflict_groups.csv"

FINAL_JSON_PATH = REPORT_DIR / "dataset_v1_final_audit.json"
FINAL_CSV_PATH = REPORT_DIR / "dataset_v1_final_audit.csv"


# ============================================================================
# IMAGE EXTENSIONS
# ============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================================
# REQUIRED COLUMNS
# ============================================================================

REQUIRED_COLUMNS = {
    "image_path",
    "unified_class",
    "split",
    "leakage_group",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def safe_int(value):
    """
    Convert Pandas / NumPy integer-like values to native Python int.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_float(value):
    """
    Convert numeric values to native Python float.
    """
    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def make_json_serializable(obj):
    """
    Recursively convert Pandas / NumPy values to native Python values.

    This fixes errors such as:

        TypeError:
        Object of type bool is not JSON serializable
    """

    # Native Python values
    if obj is None:
        return None

    if isinstance(obj, str):
        return obj

    if isinstance(obj, bool):
        return obj

    if isinstance(obj, int):
        return obj

    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None

    # Dictionaries
    if isinstance(obj, dict):
        return {
            str(key): make_json_serializable(value)
            for key, value in obj.items()
        }

    # Lists / tuples / sets
    if isinstance(obj, (list, tuple, set)):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    # NumPy / Pandas scalar types
    try:
        import numpy as np

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            value = float(obj)

            if math.isfinite(value):
                return value

            return None

    except ImportError:
        pass

    # Pandas NA / Timestamp etc.
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    try:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass

    # Final fallback
    return str(obj)


def resolve_raw_path(image_path):
    """
    Convert a manifest-relative image path into an absolute path.

    Example:

        ewaste/train/Battery/battery_106.jpg

    becomes:

        PROJECT_ROOT/data/raw/ewaste/train/Battery/battery_106.jpg
    """

    if pd.isna(image_path):
        return None

    path = Path(str(image_path))

    if path.is_absolute():
        return path

    return RAW_DIR / path


def load_manifest(path):
    """
    Load a CSV manifest safely.
    """

    if not path.exists():
        print(f"ERROR: Manifest not found:")
        print(f"  {path}")
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR: Could not read:")
        print(f"  {path}")
        print(f"Reason: {exc}")
        sys.exit(1)

    return df


# ============================================================================
# MAIN AUDIT
# ============================================================================

def main():

    print_header("WASTE SEGREGATION AI")
    print("FINAL DATASET V1 AUDIT")
    print("=" * 75)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Manifest:")
    print(MANIFEST_PATH)

    print()
    print("Curated manifest:")
    print(CURATED_MANIFEST_PATH)

    # ------------------------------------------------------------------------
    # LOAD MANIFEST
    # ------------------------------------------------------------------------

    if CURATED_MANIFEST_PATH.exists():
        print()
        print("Using curated manifest:")
        print(CURATED_MANIFEST_PATH)

        df = load_manifest(CURATED_MANIFEST_PATH)

    else:
        print()
        print("Curated manifest not found.")
        print("Using dataset_v1_manifest.csv")

        df = load_manifest(MANIFEST_PATH)

    # Keep a copy of the full dataset
    curated_df = df.copy()

    # ------------------------------------------------------------------------
    # REQUIRED COLUMN CHECK
    # ------------------------------------------------------------------------

    print_header("REQUIRED COLUMN CHECK")

    missing_columns = sorted(
        REQUIRED_COLUMNS - set(df.columns)
    )

    if missing_columns:
        print("FAIL: Missing required columns:")

        for column in missing_columns:
            print(f"  - {column}")

        required_columns_pass = False

    else:
        print("✓ All required columns present")
        required_columns_pass = True

    # ------------------------------------------------------------------------
    # BASIC DATASET INFORMATION
    # ------------------------------------------------------------------------

    print_header("DATASET SUMMARY")

    total_images = len(df)

    print(f"Total images: {total_images:,}")

    if "unified_class" in df.columns:

        class_counts = (
            df["unified_class"]
            .value_counts()
            .sort_values(ascending=False)
        )

        print()
        print(f"Number of classes: {len(class_counts)}")

        print()
        print("CLASS DISTRIBUTION")

        for class_name, count in class_counts.items():

            percentage = (
                count / total_images * 100
                if total_images > 0
                else 0
            )

            print(
                f"  {str(class_name):32s}"
                f"{count:6,d}"
                f" ({percentage:6.2f}%)"
            )

    # ------------------------------------------------------------------------
    # SPLIT DISTRIBUTION
    # ------------------------------------------------------------------------

    print_header("SPLIT × CLASS DISTRIBUTION")

    if "split" in df.columns and "unified_class" in df.columns:

        split_class_table = pd.crosstab(
            df["unified_class"],
            df["split"]
        )

        print(split_class_table.to_string())

        print()

        expected_splits = ["train", "val", "test"]

        for split in expected_splits:

            if split not in split_class_table.columns:
                print(
                    f"WARNING: '{split}' split is missing."
                )

    # ------------------------------------------------------------------------
    # MISSING VALUE CHECK
    # ------------------------------------------------------------------------

    print_header("MISSING VALUE CHECK")

    missing_required_values = {}

    for column in REQUIRED_COLUMNS:

        if column in df.columns:

            count = safe_int(
                df[column].isna().sum()
            )

            missing_required_values[column] = count

            if count > 0:
                print(
                    f"WARNING: {column}: "
                    f"{count:,} missing"
                )

    if sum(missing_required_values.values()) == 0:

        print("✓ No missing values in required fields")

        missing_required_values_pass = True

    else:

        missing_required_values_pass = False

    # ------------------------------------------------------------------------
    # IMAGE PATH DUPLICATE CHECK
    # ------------------------------------------------------------------------

    print_header("IMAGE PATH DUPLICATE CHECK")

    duplicate_path_count = 0

    if "image_path" in df.columns:

        duplicate_path_count = safe_int(
            df["image_path"].duplicated().sum()
        )

    if duplicate_path_count == 0:

        print("✓ No duplicate image paths")
        duplicate_paths_pass = True

    else:

        print(
            f"WARNING: {duplicate_path_count:,} "
            f"duplicate image paths"
        )

        duplicate_paths_pass = False

    # ------------------------------------------------------------------------
    # LEAKAGE GROUP AUDIT
    # ------------------------------------------------------------------------

    print_header("LEAKAGE GROUP AUDIT")

    leakage_group_count = 0
    grouped_group_count = 0
    grouped_image_count = 0
    cross_split_group_count = 0
    maximum_group_size = 0

    if "leakage_group" in df.columns:

        group_sizes = (
            df.groupby("leakage_group")
            .size()
        )

        leakage_group_count = len(group_sizes)

        grouped_groups = group_sizes[
            group_sizes > 1
        ]

        grouped_group_count = len(grouped_groups)

        grouped_image_count = safe_int(
            grouped_groups.sum()
        )

        maximum_group_size = (
            safe_int(group_sizes.max())
            if len(group_sizes) > 0
            else 0
        )

        print(
            f"Total leakage groups: "
            f"{leakage_group_count:,}"
        )

        print(
            f"Groups containing >1 image: "
            f"{grouped_group_count:,}"
        )

        print(
            f"Images belonging to grouped samples: "
            f"{grouped_image_count:,}"
        )

        # ------------------------------------------------------------
        # CROSS-SPLIT LEAKAGE
        # ------------------------------------------------------------

        print()
        print(
            "Checking leakage groups crossing splits..."
        )

        if "split" in df.columns:

            group_split_counts = (
                df.groupby("leakage_group")["split"]
                .nunique()
            )

            cross_split_groups = group_split_counts[
                group_split_counts > 1
            ]

            cross_split_group_count = len(
                cross_split_groups
            )

        if cross_split_group_count == 0:

            print(
                "✓ No leakage groups cross "
                "train/val/test"
            )

            cross_split_leakage_pass = True

        else:

            print(
                f"WARNING: {cross_split_group_count:,} "
                f"groups cross splits"
            )

            cross_split_leakage_pass = False

    else:

        print("WARNING: leakage_group column missing")
        cross_split_leakage_pass = False

    # ------------------------------------------------------------------------
    # LABEL CONFLICT AUDIT
    # ------------------------------------------------------------------------

    print_header("LABEL CONFLICT AUDIT")

    conflict_group_count = 0
    conflict_distribution = {}
    conflicting_group_names = []

    if "leakage_group" in df.columns and "unified_class" in df.columns:

        class_per_group = (
            df.groupby("leakage_group")["unified_class"]
            .nunique()
        )

        conflicting_groups = class_per_group[
            class_per_group > 1
        ]

        conflict_group_count = len(
            conflicting_groups
        )

        conflict_distribution = (
            conflicting_groups
            .value_counts()
            .sort_index()
            .to_dict()
        )

        conflicting_group_names = [
            str(x)
            for x in conflicting_groups.index.tolist()
        ]

        print(
            f"Leakage groups: "
            f"{leakage_group_count:,}"
        )

        print(
            f"Conflicting groups: "
            f"{conflict_group_count:,}"
        )

        if conflict_group_count > 0:

            print()
            print("Conflict distribution:")

            for class_count, group_count in (
                conflict_distribution.items()
            ):

                print(
                    f"  {class_count} classes: "
                    f"{group_count} groups"
                )

            print()
            print(
                "WARNING: Conflicting groups remain:"
            )

            print(
                conflicting_groups
                .sort_values(ascending=False)
                .head(20)
                .to_string()
            )

        else:

            print(
                "✓ No conflicting leakage groups"
            )

    # ------------------------------------------------------------------------
    # EXACT DUPLICATE AUDIT
    # ------------------------------------------------------------------------

    print_header("EXACT DUPLICATE CHECK")

    exact_images = 0
    exact_groups = 0

    if EXACT_DUPLICATES_PATH.exists():

        try:

            exact_df = pd.read_csv(
                EXACT_DUPLICATES_PATH
            )

            if "duplicate_group" in exact_df.columns:

                exact_images = len(exact_df)

                exact_groups = (
                    exact_df["duplicate_group"]
                    .nunique()
                )

        except Exception as exc:

            print(
                f"WARNING: Could not read exact "
                f"duplicate report: {exc}"
            )

    print(
        f"Images assigned exact duplicate groups: "
        f"{exact_images:,}"
    )

    print(
        f"Exact duplicate groups represented: "
        f"{exact_groups:,}"
    )

    # ------------------------------------------------------------------------
    # NEAR DUPLICATE AUDIT
    # ------------------------------------------------------------------------

    print_header("NEAR-DUPLICATE CHECK")

    near_images = 0
    near_groups = 0

    if NEAR_DUPLICATES_PATH.exists():

        try:

            near_df = pd.read_csv(
                NEAR_DUPLICATES_PATH
            )

            if {
                "path_a",
                "path_b"
            }.issubset(near_df.columns):

                near_paths = set(
                    near_df["path_a"].dropna()
                )

                near_paths.update(
                    near_df["path_b"].dropna()
                )

                near_images = len(near_paths)

                # Number of connected components is not
                # required here. We report unique images
                # participating in near duplicates.

                near_groups = len(near_df)

            elif "near_duplicate_group" in near_df.columns:

                near_images = len(near_df)

                near_groups = (
                    near_df[
                        "near_duplicate_group"
                    ].nunique()
                )

            else:

                near_images = len(near_df)

        except Exception as exc:

            print(
                f"WARNING: Could not read near "
                f"duplicate report: {exc}"
            )

    print(
        f"Images assigned near-duplicate groups: "
        f"{near_images:,}"
    )

    print(
        f"Near-duplicate pairs/groups represented: "
        f"{near_groups:,}"
    )

    # ------------------------------------------------------------------------
    # IMAGE FILE EXISTENCE CHECK
    # ------------------------------------------------------------------------

    print_header("IMAGE FILE EXISTENCE CHECK")

    print(
        "Checking curated image paths against data/raw..."
    )

    existing_images = 0
    missing_images = 0
    missing_image_paths = []

    if "image_path" in df.columns:

        for image_path in df["image_path"]:

            resolved = resolve_raw_path(
                image_path
            )

            if (
                resolved is not None
                and resolved.exists()
                and resolved.is_file()
            ):

                existing_images += 1

            else:

                missing_images += 1

                if len(missing_image_paths) < 50:
                    missing_image_paths.append(
                        str(image_path)
                    )

    print(
        f"Images found: "
        f"{existing_images:,}/{total_images:,}"
    )

    print(
        f"Images missing: "
        f"{missing_images:,}"
    )

    if missing_images == 0:

        print("✓ All curated image files exist")
        missing_images_pass = True

    else:

        print()
        print("First missing images:")

        for path in missing_image_paths:
            print(f"  {path}")

        missing_images_pass = False

    # ------------------------------------------------------------------------
    # CLASS PRESENCE VALIDATION
    # ------------------------------------------------------------------------

    print_header("CLASS PRESENCE VALIDATION")

    train_classes = set()
    val_classes = set()
    test_classes = set()
    all_classes = set()

    if (
        "unified_class" in df.columns
        and "split" in df.columns
    ):

        all_classes = set(
            df["unified_class"]
            .dropna()
            .astype(str)
            .unique()
        )

        train_classes = set(
            df.loc[
                df["split"].astype(str).str.lower()
                == "train",
                "unified_class"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        val_classes = set(
            df.loc[
                df["split"].astype(str).str.lower()
                == "val",
                "unified_class"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        test_classes = set(
            df.loc[
                df["split"].astype(str).str.lower()
                == "test",
                "unified_class"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

    missing_train_classes = sorted(
        all_classes - train_classes
    )

    missing_val_classes = sorted(
        all_classes - val_classes
    )

    missing_test_classes = sorted(
        all_classes - test_classes
    )

    if not missing_train_classes:

        print(
            "✓ Every class has training samples"
        )

        all_classes_train_pass = True

    else:

        print(
            "WARNING: Classes missing from train:"
        )

        for class_name in missing_train_classes:
            print(f"  - {class_name}")

        all_classes_train_pass = False

    if not missing_val_classes:

        print(
            "✓ Every class has validation samples"
        )

        all_classes_val_pass = True

    else:

        print(
            "WARNING: Classes missing from val:"
        )

        for class_name in missing_val_classes:
            print(f"  - {class_name}")

        all_classes_val_pass = False

    if not missing_test_classes:

        print(
            "✓ Every class has test samples"
        )

        all_classes_test_pass = True

    else:

        print(
            "WARNING: Classes missing from test:"
        )

        for class_name in missing_test_classes:
            print(f"  - {class_name}")

        all_classes_test_pass = False

    # ------------------------------------------------------------------------
    # DATASET RETENTION
    # ------------------------------------------------------------------------

    print_header("DATASET RETENTION")

    # IMPORTANT:
    # We compare the curated manifest with the original
    # dataset_v1 manifest, NOT with TrashNet's 2,527 images.
    #
    # This fixes the previous incorrect result:
    #
    #     2,527 -> 13,196 -> 522.20%
    #
    # The combined dataset contains 13,196 images.

    original_manifest_count = None

    if MANIFEST_PATH.exists():

        try:

            original_manifest_df = pd.read_csv(
                MANIFEST_PATH
            )

            original_manifest_count = len(
                original_manifest_df
            )

        except Exception:
            original_manifest_count = None

    if original_manifest_count is None:

        original_manifest_count = total_images

    curated_image_count = total_images

    images_excluded = max(
        0,
        original_manifest_count - curated_image_count
    )

    if original_manifest_count > 0:

        retention_rate = (
            curated_image_count
            / original_manifest_count
        ) * 100

    else:

        retention_rate = 0.0

    print(
        f"Original dataset image count: "
        f"{original_manifest_count:,}"
    )

    print(
        f"Curated image count: "
        f"{curated_image_count:,}"
    )

    print(
        f"Images excluded: "
        f"{images_excluded:,}"
    )

    print(
        f"Retention rate: "
        f"{retention_rate:.2f}%"
    )

    # ------------------------------------------------------------------------
    # SPLIT COUNTS
    # ------------------------------------------------------------------------

    train_count = 0
    val_count = 0
    test_count = 0

    if "split" in df.columns:

        split_counts = (
            df["split"]
            .astype(str)
            .str.lower()
            .value_counts()
        )

        train_count = safe_int(
            split_counts.get("train", 0)
        )

        val_count = safe_int(
            split_counts.get("val", 0)
        )

        test_count = safe_int(
            split_counts.get("test", 0)
        )

    # ------------------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------------------

    print_header("FINAL DATASET STATUS")

    checks = {
        "required_columns": required_columns_pass,
        "missing_required_values":
            missing_required_values_pass,
        "duplicate_paths":
            duplicate_paths_pass,
        "cross_split_leakage":
            cross_split_leakage_pass,
        "missing_images":
            missing_images_pass,
        "all_classes_train":
            all_classes_train_pass,
        "all_classes_val":
            all_classes_val_pass,
        "all_classes_test":
            all_classes_test_pass,
    }

    for name, passed in checks.items():

        if bool(passed):
            print(f"  [PASS] {name}")

        else:
            print(f"  [FAIL] {name}")

    # Label conflicts are a warning, not an automatic
    # failure because the dataset may contain genuinely
    # ambiguous visual examples.

    if conflict_group_count > 0:

        print(
            "  [WARN] label_conflicts"
        )

    else:

        print(
            "  [PASS] label_conflicts"
        )

    integrity_pass = all(
        bool(value)
        for value in checks.values()
    )

    if integrity_pass:

        print()
        print(
            "✓ DATASET V1 PASSED FINAL INTEGRITY CHECK"
        )

    else:

        print()
        print(
            "✗ DATASET V1 FAILED FINAL INTEGRITY CHECK"
        )

    # ------------------------------------------------------------------------
    # CREATE AUDIT DATA
    # ------------------------------------------------------------------------

    print_header("CREATING AUDIT REPORT")

    audit_results = {

        "dataset": {
            "name": "Waste Segregation AI Dataset V1",
            "total_images": total_images,
            "number_of_classes": len(all_classes),
            "classes": sorted(all_classes),
        },

        "splits": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": (
                train_count
                + val_count
                + test_count
            ),
        },

        "class_distribution": {
            str(class_name): safe_int(count)
            for class_name, count
            in (
                df["unified_class"]
                .value_counts()
                .items()
            )
        },

        "leakage": {
            "total_groups":
                leakage_group_count,

            "groups_containing_multiple_images":
                grouped_group_count,

            "images_in_grouped_samples":
                grouped_image_count,

            "maximum_group_size":
                maximum_group_size,

            "groups_crossing_splits":
                cross_split_group_count,
        },

        "duplicates": {

            "exact_duplicate_images":
                exact_images,

            "exact_duplicate_groups":
                exact_groups,

            "near_duplicate_images":
                near_images,

            "near_duplicate_pairs_or_groups":
                near_groups,
        },

        "label_conflicts": {

            "conflicting_groups":
                conflict_group_count,

            "conflict_distribution":
                conflict_distribution,

            "conflicting_group_names":
                conflicting_group_names,
        },

        "files": {

            "images_expected":
                total_images,

            "images_found":
                existing_images,

            "images_missing":
                missing_images,
        },

        "retention": {

            "original_manifest_images":
                original_manifest_count,

            "curated_images":
                curated_image_count,

            "images_excluded":
                images_excluded,

            "retention_rate_percent":
                retention_rate,
        },

        "validation": {

            "required_columns":
                required_columns_pass,

            "missing_required_values":
                missing_required_values_pass,

            "duplicate_paths":
                duplicate_paths_pass,

            "cross_split_leakage":
                cross_split_leakage_pass,

            "missing_images":
                missing_images_pass,

            "all_classes_train":
                all_classes_train_pass,

            "all_classes_val":
                all_classes_val_pass,

            "all_classes_test":
                all_classes_test_pass,

            "overall_integrity":
                integrity_pass,

            "label_conflicts_warning":
                conflict_group_count > 0,
        },
    }

    # ------------------------------------------------------------------------
    # JSON REPORT
    # ------------------------------------------------------------------------

    json_safe_audit = make_json_serializable(
        audit_results
    )

    try:

        with open(
            FINAL_JSON_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                json_safe_audit,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"✓ {FINAL_JSON_PATH}"
        )

    except Exception as exc:

        print()
        print(
            "ERROR: Could not create JSON report:"
        )

        print(exc)

        sys.exit(1)

    # ------------------------------------------------------------------------
    # FLAT CSV REPORT
    # ------------------------------------------------------------------------

    csv_rows = [

        {
            "category": "dataset",
            "metric": "total_images",
            "value": total_images,
        },

        {
            "category": "dataset",
            "metric": "number_of_classes",
            "value": len(all_classes),
        },

        {
            "category": "split",
            "metric": "train",
            "value": train_count,
        },

        {
            "category": "split",
            "metric": "val",
            "value": val_count,
        },

        {
            "category": "split",
            "metric": "test",
            "value": test_count,
        },

        {
            "category": "leakage",
            "metric": "total_groups",
            "value": leakage_group_count,
        },

        {
            "category": "leakage",
            "metric": "groups_multiple_images",
            "value": grouped_group_count,
        },

        {
            "category": "leakage",
            "metric": "grouped_images",
            "value": grouped_image_count,
        },

        {
            "category": "leakage",
            "metric": "cross_split_groups",
            "value": cross_split_group_count,
        },

        {
            "category": "duplicates",
            "metric": "exact_duplicate_images",
            "value": exact_images,
        },

        {
            "category": "duplicates",
            "metric": "exact_duplicate_groups",
            "value": exact_groups,
        },

        {
            "category": "duplicates",
            "metric": "near_duplicate_images",
            "value": near_images,
        },

        {
            "category": "duplicates",
            "metric": "near_duplicate_pairs_or_groups",
            "value": near_groups,
        },

        {
            "category": "label_conflicts",
            "metric": "conflicting_groups",
            "value": conflict_group_count,
        },

        {
            "category": "files",
            "metric": "images_expected",
            "value": total_images,
        },

        {
            "category": "files",
            "metric": "images_found",
            "value": existing_images,
        },

        {
            "category": "files",
            "metric": "images_missing",
            "value": missing_images,
        },

        {
            "category": "retention",
            "metric": "original_manifest_images",
            "value": original_manifest_count,
        },

        {
            "category": "retention",
            "metric": "curated_images",
            "value": curated_image_count,
        },

        {
            "category": "retention",
            "metric": "images_excluded",
            "value": images_excluded,
        },

        {
            "category": "retention",
            "metric": "retention_rate_percent",
            "value": retention_rate,
        },

        {
            "category": "validation",
            "metric": "required_columns",
            "value": bool(required_columns_pass),
        },

        {
            "category": "validation",
            "metric": "missing_required_values",
            "value": bool(
                missing_required_values_pass
            ),
        },

        {
            "category": "validation",
            "metric": "duplicate_paths",
            "value": bool(
                duplicate_paths_pass
            ),
        },

        {
            "category": "validation",
            "metric": "cross_split_leakage",
            "value": bool(
                cross_split_leakage_pass
            ),
        },

        {
            "category": "validation",
            "metric": "missing_images",
            "value": bool(
                missing_images_pass
            ),
        },

        {
            "category": "validation",
            "metric": "all_classes_train",
            "value": bool(
                all_classes_train_pass
            ),
        },

        {
            "category": "validation",
            "metric": "all_classes_val",
            "value": bool(
                all_classes_val_pass
            ),
        },

        {
            "category": "validation",
            "metric": "all_classes_test",
            "value": bool(
                all_classes_test_pass
            ),
        },

        {
            "category": "validation",
            "metric": "overall_integrity",
            "value": bool(integrity_pass),
        },

        {
            "category": "validation",
            "metric": "label_conflicts_warning",
            "value": bool(
                conflict_group_count > 0
            ),
        },
    ]

    audit_csv_df = pd.DataFrame(csv_rows)

    try:

        audit_csv_df.to_csv(
            FINAL_CSV_PATH,
            index=False
        )

        print(
            f"✓ {FINAL_CSV_PATH}"
        )

    except Exception as exc:

        print()
        print(
            "ERROR: Could not create CSV report:"
        )

        print(exc)

        sys.exit(1)

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------

    print_header("AUDIT COMPLETE")

    print(
        f"Dataset images:       {total_images:,}"
    )

    print(
        f"Train:                 {train_count:,}"
    )

    print(
        f"Validation:            {val_count:,}"
    )

    print(
        f"Test:                  {test_count:,}"
    )

    print(
        f"Classes:               {len(all_classes)}"
    )

    print(
        f"Leakage groups:        {leakage_group_count:,}"
    )

    print(
        f"Cross-split leakage:   {cross_split_group_count:,}"
    )

    print(
        f"Exact duplicate imgs:  {exact_images:,}"
    )

    print(
        f"Near duplicate imgs:   {near_images:,}"
    )

    print(
        f"Label conflicts:       {conflict_group_count:,}"
    )

    print(
        f"Missing images:        {missing_images:,}"
    )

    print(
        f"Retention:             {retention_rate:.2f}%"
    )

    print()

    if integrity_pass:

        print(
            "STATUS: PASS"
        )

        print(
            "Dataset is ready for the model-training stage."
        )

    else:

        print(
            "STATUS: FAIL"
        )

        print(
            "Fix the failed checks before training."
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()