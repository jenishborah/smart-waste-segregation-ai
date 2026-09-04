"""
Waste Segregation AI
====================

Duplicate Detector

Purpose:
    Detect exact and visually similar duplicate images across
    all locally stored waste datasets.

Datasets:
    - TrashNet
    - RealWaste
    - PhenomSG
    - E-Waste

IMPORTANT:
    This script is READ-ONLY.

    It does NOT:
        - delete images
        - move images
        - rename images
        - resize images
        - modify datasets

It only creates duplicate reports.

Detection methods:
    1. SHA-256 hash       -> exact duplicate detection
    2. Perceptual hash    -> near-duplicate detection
"""

from pathlib import Path
from collections import defaultdict
import hashlib
import csv
import time

from PIL import Image
import imagehash


# ============================================================
# PROJECT PATHS
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


# ============================================================
# PERCEPTUAL HASH CONFIGURATION
# ============================================================

# pHash produces a 64-bit hash by default.
#
# Smaller Hamming distance = more visually similar.
#
# <= 6:
#     Strong near-duplicate candidate
#
# 7-10:
#     Possible similarity, requires manual review
#
# We will report both groups separately.

STRONG_DUPLICATE_THRESHOLD = 6
REVIEW_THRESHOLD = 10


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(dataset_path):
    """
    Recursively find supported image files.
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
# SHA-256
# ============================================================

def calculate_sha256(file_path):
    """
    Calculate SHA-256 hash of the original file.

    This detects exact file duplicates.
    """

    sha256 = hashlib.sha256()

    try:

        with file_path.open("rb") as file:

            while True:

                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception:
        return None


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def calculate_phash(file_path):
    """
    Calculate perceptual hash.

    Perceptual hashing allows us to identify images that look
    similar even when the files are not byte-for-byte identical.
    """

    try:

        with Image.open(file_path) as image:

            # Convert to RGB for consistency
            image = image.convert("RGB")

            return str(
                imagehash.phash(image)
            )

    except Exception:
        return None


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(hash_a, hash_b):
    """
    Calculate Hamming distance between two hexadecimal
    perceptual hashes.
    """

    if hash_a is None or hash_b is None:
        return None

    try:

        binary_a = bin(
            int(hash_a, 16)
        )[2:].zfill(len(hash_a) * 4)

        binary_b = bin(
            int(hash_b, 16)
        )[2:].zfill(len(hash_b) * 4)

        return sum(
            bit_a != bit_b
            for bit_a, bit_b in zip(
                binary_a,
                binary_b
            )
        )

    except Exception:
        return None


# ============================================================
# DATASET COLLECTION
# ============================================================

def collect_images():

    all_images = []

    print("\n")
    print("=" * 75)
    print("SCANNING DATASETS")
    print("=" * 75)

    for dataset_name, dataset_path in DATASETS.items():

        print(
            f"\n{dataset_name}:"
        )

        if not dataset_path.exists():

            print(
                f"  [NOT FOUND] {dataset_path}"
            )

            continue

        images = find_images(
            dataset_path
        )

        print(
            f"  Images found: {len(images):,}"
        )

        for image_path in images:

            all_images.append({
                "dataset": dataset_name,
                "path": image_path,
                "relative_path": str(
                    image_path.relative_to(
                        dataset_path
                    )
                ),
            })

    print(
        f"\nTotal images discovered: "
        f"{len(all_images):,}"
    )

    return all_images


# ============================================================
# EXACT DUPLICATE DETECTION
# ============================================================

def find_exact_duplicates(all_images):

    print("\n")
    print("=" * 75)
    print("EXACT DUPLICATE DETECTION")
    print("=" * 75)

    hash_groups = defaultdict(list)

    total = len(all_images)

    start_time = time.time()

    for index, item in enumerate(
        all_images,
        start=1
    ):

        file_hash = calculate_sha256(
            item["path"]
        )

        item["sha256"] = file_hash

        if file_hash:

            hash_groups[file_hash].append(
                item
            )

        if index % 500 == 0:

            print(
                f"  Processed "
                f"{index:,}/{total:,}"
            )

    duplicate_groups = []

    for file_hash, items in hash_groups.items():

        if len(items) > 1:

            duplicate_groups.append({
                "hash": file_hash,
                "items": items,
            })

    elapsed = time.time() - start_time

    duplicate_image_count = sum(
        len(group["items"])
        for group in duplicate_groups
    )

    print(
        f"\nExact duplicate groups: "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Images involved: "
        f"{duplicate_image_count:,}"
    )

    print(
        f"Time: {elapsed:.2f} seconds"
    )

    return duplicate_groups


# ============================================================
# CREATE PHASH INDEX
# ============================================================

def create_phash_index(all_images):

    print("\n")
    print("=" * 75)
    print("CALCULATING PERCEPTUAL HASHES")
    print("=" * 75)

    phash_groups = defaultdict(list)

    total = len(all_images)

    start_time = time.time()

    successful = 0
    failed = 0

    for index, item in enumerate(
        all_images,
        start=1
    ):

        phash = calculate_phash(
            item["path"]
        )

        item["phash"] = phash

        if phash:

            phash_groups[phash].append(
                item
            )

            successful += 1

        else:

            failed += 1

        if index % 500 == 0:

            print(
                f"  Processed "
                f"{index:,}/{total:,}"
            )

    elapsed = time.time() - start_time

    print(
        f"\nSuccessful hashes: {successful:,}"
    )

    print(
        f"Failed hashes:     {failed:,}"
    )

    print(
        f"Time: {elapsed:.2f} seconds"
    )

    return phash_groups


# ============================================================
# CREATE HASH BUCKETS
# ============================================================

def create_buckets(all_images):

    """
    Create multiple buckets from different parts of the
    perceptual hash.

    This avoids comparing every image with every other image.

    A 64-bit pHash is divided into four 16-bit sections.

    Two images that share a section become candidate pairs.
    """

    buckets = [
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    ]

    for item in all_images:

        phash = item.get("phash")

        if not phash:
            continue

        # pHash is normally 16 hexadecimal characters.
        # Divide into four groups of four hex characters.
        sections = [
            phash[0:4],
            phash[4:8],
            phash[8:12],
            phash[12:16],
        ]

        for index, section in enumerate(
            sections
        ):

            buckets[index][section].append(
                item
            )

    return buckets


# ============================================================
# NEAR DUPLICATE DETECTION
# ============================================================

def find_near_duplicates(all_images):

    print("\n")
    print("=" * 75)
    print("NEAR DUPLICATE DETECTION")
    print("=" * 75)

    buckets = create_buckets(
        all_images
    )

    # Use object IDs to prevent duplicate candidate pairs.
    candidate_pairs = set()

    # --------------------------------------------------------
    # Generate candidate pairs
    # --------------------------------------------------------

    for bucket in buckets:

        for items in bucket.values():

            if len(items) < 2:
                continue

            # Compare only within the bucket.
            for i in range(len(items)):

                for j in range(
                    i + 1,
                    len(items)
                ):

                    path_a = str(
                        items[i]["path"]
                    )

                    path_b = str(
                        items[j]["path"]
                    )

                    pair = tuple(
                        sorted(
                            [path_a, path_b]
                        )
                    )

                    candidate_pairs.add(
                        pair
                    )

    print(
        f"Candidate pairs: "
        f"{len(candidate_pairs):,}"
    )

    strong_matches = []
    review_matches = []

    # Map paths to image records
    image_lookup = {
        str(item["path"]): item
        for item in all_images
    }

    start_time = time.time()

    # --------------------------------------------------------
    # Compare candidate pairs
    # --------------------------------------------------------

    for index, pair in enumerate(
        candidate_pairs,
        start=1
    ):

        item_a = image_lookup[
            pair[0]
        ]

        item_b = image_lookup[
            pair[1]
        ]

        distance = hamming_distance(
            item_a.get("phash"),
            item_b.get("phash")
        )

        if distance is None:
            continue

        match = {
            "distance": distance,
            "dataset_a": item_a["dataset"],
            "file_a": item_a["relative_path"],
            "path_a": str(
                item_a["path"]
            ),
            "dataset_b": item_b["dataset"],
            "file_b": item_b["relative_path"],
            "path_b": str(
                item_b["path"]
            ),
        }

        if distance <= STRONG_DUPLICATE_THRESHOLD:

            strong_matches.append(
                match
            )

        elif distance <= REVIEW_THRESHOLD:

            review_matches.append(
                match
            )

        if index % 5000 == 0:

            print(
                f"  Compared "
                f"{index:,}/{len(candidate_pairs):,}"
            )

    elapsed = time.time() - start_time

    print(
        f"\nStrong near-duplicates: "
        f"{len(strong_matches):,}"
    )

    print(
        f"Review candidates: "
        f"{len(review_matches):,}"
    )

    print(
        f"Time: {elapsed:.2f} seconds"
    )

    return (
        strong_matches,
        review_matches,
    )


# ============================================================
# WRITE EXACT DUPLICATE REPORT
# ============================================================

def write_exact_duplicate_report(
    duplicate_groups
):

    rows = []

    group_id = 1

    for group in duplicate_groups:

        for item in group["items"]:

            rows.append({
                "duplicate_group": group_id,
                "sha256": group["hash"],
                "dataset": item["dataset"],
                "file": item["relative_path"],
                "absolute_path": str(
                    item["path"]
                ),
            })

        group_id += 1

    output = (
        REPORT_DIR /
        "exact_duplicates.csv"
    )

    write_csv(
        output,
        [
            "duplicate_group",
            "sha256",
            "dataset",
            "file",
            "absolute_path",
        ],
        rows
    )


# ============================================================
# WRITE NEAR DUPLICATE REPORT
# ============================================================
def write_near_duplicate_report(matches, filename):
    """
    Write near-duplicate matches to CSV.

    Internally the detector stores the similarity value as
    'distance'. The CSV uses the clearer name
    'hamming_distance'.
    """

    output = REPORT_DIR / filename

    rows = []

    for match in matches:
        rows.append({
            "hamming_distance": match["distance"],
            "dataset_a": match["dataset_a"],
            "file_a": match["file_a"],
            "path_a": match["path_a"],
            "dataset_b": match["dataset_b"],
            "file_b": match["file_b"],
            "path_b": match["path_b"],
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
        rows
    )
# ============================================================
# CSV WRITER
# ============================================================

def write_csv(
    file_path,
    headers,
    rows
):

    REPORT_DIR.mkdir(
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
# SUMMARY REPORT
# ============================================================

def write_summary(
    all_images,
    exact_groups,
    strong_matches,
    review_matches
):

    exact_images = sum(
        len(group["items"])
        for group in exact_groups
    )

    cross_dataset_exact = 0

    for group in exact_groups:

        datasets = {
            item["dataset"]
            for item in group["items"]
        }

        if len(datasets) > 1:

            cross_dataset_exact += 1

    cross_dataset_strong = sum(
        1
        for match in strong_matches
        if match["dataset_a"]
        != match["dataset_b"]
    )

    cross_dataset_review = sum(
        1
        for match in review_matches
        if match["dataset_a"]
        != match["dataset_b"]
    )

    rows = [
        {
            "metric": "total_images_scanned",
            "value": len(all_images),
        },
        {
            "metric": "exact_duplicate_groups",
            "value": len(exact_groups),
        },
        {
            "metric": "images_in_exact_duplicate_groups",
            "value": exact_images,
        },
        {
            "metric": "cross_dataset_exact_duplicate_groups",
            "value": cross_dataset_exact,
        },
        {
            "metric": "strong_near_duplicate_pairs",
            "value": len(strong_matches),
        },
        {
            "metric": "cross_dataset_strong_near_duplicates",
            "value": cross_dataset_strong,
        },
        {
            "metric": "review_similarity_pairs",
            "value": len(review_matches),
        },
        {
            "metric": "cross_dataset_review_pairs",
            "value": cross_dataset_review,
        },
    ]

    write_csv(
        REPORT_DIR / "duplicate_summary.csv",
        [
            "metric",
            "value",
        ],
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("WASTE SEGREGATION AI")
    print("DUPLICATE DETECTOR")
    print("=" * 75)

    print(
        "\nRaw dataset directory:"
    )

    print(RAW_DATA_DIR)

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    all_images = collect_images()

    if not all_images:

        print(
            "\nERROR: No images found."
        )

        return

    # --------------------------------------------------------
    # Exact duplicates
    # --------------------------------------------------------

    exact_groups = find_exact_duplicates(
        all_images
    )

    # --------------------------------------------------------
    # Perceptual hashes
    # --------------------------------------------------------

    create_phash_index(
        all_images
    )

    # --------------------------------------------------------
    # Near duplicates
    # --------------------------------------------------------

    (
        strong_matches,
        review_matches,
    ) = find_near_duplicates(
        all_images
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    write_exact_duplicate_report(
        exact_groups
    )

    write_near_duplicate_report(
        strong_matches,
        "near_duplicates_strong.csv"
    )

    write_near_duplicate_report(
        review_matches,
        "near_duplicates_review.csv"
    )

    write_summary(
        all_images,
        exact_groups,
        strong_matches,
        review_matches
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 75)
    print("DUPLICATE DETECTION COMPLETE")
    print("=" * 75)

    print(
        f"\nImages scanned: "
        f"{len(all_images):,}"
    )

    print(
        f"Exact duplicate groups: "
        f"{len(exact_groups):,}"
    )

    print(
        f"Strong near-duplicates: "
        f"{len(strong_matches):,}"
    )

    print(
        f"Review candidates: "
        f"{len(review_matches):,}"
    )

    print(
        "\nReports created:"
    )

    print(
        "  duplicate_summary.csv"
    )

    print(
        "  exact_duplicates.csv"
    )

    print(
        "  near_duplicates_strong.csv"
    )

    print(
        "  near_duplicates_review.csv"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No dataset files were modified."
    )

    print(
        "No images were deleted."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()