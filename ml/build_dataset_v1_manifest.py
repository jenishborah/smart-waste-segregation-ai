from pathlib import Path
from collections import defaultdict
import hashlib
import random

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path("data/raw")
REPORT_ROOT = Path("data/reports")

TAXONOMY_FILE = REPORT_ROOT / "taxonomy_mapping.csv"
EXACT_DUPLICATE_FILE = REPORT_ROOT / "exact_duplicates.csv"
NEAR_DUPLICATE_FILE = REPORT_ROOT / "near_duplicates_strong.csv"

OUTPUT_MANIFEST = REPORT_ROOT / "dataset_v1_manifest.csv"
OUTPUT_SUMMARY = REPORT_ROOT / "dataset_v1_summary.csv"
OUTPUT_GROUPS = REPORT_ROOT / "dataset_v1_duplicate_groups.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

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
# HELPER FUNCTIONS
# ============================================================

def normalize_path(path):
    """
    Normalize path representation for matching CSV paths.
    """

    return str(path).replace("\\", "/").strip()


def relative_to_raw(path):
    """
    Always return a normalized path relative to data/raw.

    Handles both:
      - absolute Windows paths
      - relative paths such as data/raw/...
      - Path objects returned by rglob()
    """

    path = Path(path)

    raw = DATA_ROOT.resolve()

    try:
        # Convert to absolute first.
        absolute = path.resolve()

        return absolute.relative_to(
            raw
        ).as_posix()

    except ValueError:

        # If the supplied path is already relative
        # to data/raw, normalize it directly.
        return path.as_posix().replace(
            "\\",
            "/"
        ).replace(
            "data/raw/",
            ""
        ).replace(
            "data\\raw\\",
            ""
        )

def find_all_images():
    """
    Find all images inside data/raw.
    """

    images = []

    for path in DATA_ROOT.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


def stable_group_id(prefix, index):
    return f"{prefix}_{index:05d}"


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):

        if item not in self.parent:

            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, a, b):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:

            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:

            self.parent[root_b] = root_a

        else:

            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# READ DUPLICATE GROUPS
# ============================================================


def build_duplicate_groups(all_images):

    print("\n" + "=" * 75)
    print("BUILDING DUPLICATE GROUPS")
    print("=" * 75)

    # --------------------------------------------------------
    # All real images as normalized relative paths
    # --------------------------------------------------------

    image_set = {
        relative_to_raw(path)
        for path in all_images
    }

    print(
        f"\nImages available for grouping: "
        f"{len(image_set):,}"
    )

    # ========================================================
    # ONE UNION-FIND FOR ALL DUPLICATE RELATIONSHIPS
    # ========================================================

    uf = UnionFind()

    for image in image_set:
        uf.add(image)

    # ========================================================
    # EXACT DUPLICATES
    # ========================================================

    exact_lookup = {}

    if EXACT_DUPLICATE_FILE.exists():

        exact_df = pd.read_csv(
            EXACT_DUPLICATE_FILE
        )

        print(
            f"\nExact duplicate rows: "
            f"{len(exact_df):,}"
        )

        # Every row belongs to a duplicate_group.
        # All files with the same duplicate_group
        # must belong to one leakage group.

        for duplicate_id, group in exact_df.groupby(
            "duplicate_group"
        ):

            members = []

            for absolute_path in group[
                "absolute_path"
            ]:

                try:

                    relative_path = relative_to_raw(
                        Path(
                            str(absolute_path).strip()
                        )
                    )

                except Exception:

                    continue

                if relative_path in image_set:

                    members.append(
                        relative_path
                    )

            # Connect every member to the first member.
            if len(members) >= 2:

                first = members[0]

                for member in members[1:]:

                    uf.union(
                        first,
                        member
                    )

    else:

        print(
            "WARNING: exact_duplicates.csv "
            "not found."
        )

    # ========================================================
    # STRONG NEAR DUPLICATES
    # ========================================================

    if NEAR_DUPLICATE_FILE.exists():

        near_df = pd.read_csv(
            NEAR_DUPLICATE_FILE
        )

        print(
            f"Strong near-duplicate rows: "
            f"{len(near_df):,}"
        )

        for _, row in near_df.iterrows():

            try:

                path_a = relative_to_raw(
                    Path(
                        str(row["path_a"]).strip()
                    )
                )

                path_b = relative_to_raw(
                    Path(
                        str(row["path_b"]).strip()
                    )
                )

            except Exception:

                continue

            if (
                path_a in image_set
                and path_b in image_set
            ):

                uf.union(
                    path_a,
                    path_b
                )

    else:

        print(
            "WARNING: near_duplicates_strong.csv "
            "not found."
        )

    # ========================================================
    # BUILD CONNECTED COMPONENTS
    # ========================================================

    groups = defaultdict(list)

    for image in image_set:

        root = uf.find(image)

        groups[root].append(
            image
        )

    # ========================================================
    # CREATE STABLE LEAKAGE GROUP IDs
    # ========================================================

    leakage_lookup = {}

    for index, members in enumerate(
        groups.values(),
        start=1,
    ):

        group_id = stable_group_id(
            "GROUP",
            index,
        )

        for image in members:

            leakage_lookup[
                image
            ] = group_id

    # ========================================================
    # CREATE EXACT DUPLICATE LOOKUP
    # ========================================================

    if EXACT_DUPLICATE_FILE.exists():

        exact_df = pd.read_csv(
            EXACT_DUPLICATE_FILE
        )

        # Map each image to its original exact group.
        for duplicate_id, group in exact_df.groupby(
            "duplicate_group"
        ):

            for absolute_path in group[
                "absolute_path"
            ]:

                try:

                    relative_path = relative_to_raw(
                        Path(
                            str(absolute_path).strip()
                        )
                    )

                except Exception:

                    continue

                if relative_path in image_set:

                    exact_lookup[
                        relative_path
                    ] = str(
                        duplicate_id
                    )

    # ========================================================
    # CREATE NEAR DUPLICATE LOOKUP
    # ========================================================

    near_lookup = {}

    if NEAR_DUPLICATE_FILE.exists():

        near_df = pd.read_csv(
            NEAR_DUPLICATE_FILE
        )

        # Give each connected near-duplicate component
        # a separate identifier.

        near_uf = UnionFind()

        for image in image_set:

            near_uf.add(image)

        for _, row in near_df.iterrows():

            try:

                path_a = relative_to_raw(
                    Path(
                        str(row["path_a"]).strip()
                    )
                )

                path_b = relative_to_raw(
                    Path(
                        str(row["path_b"]).strip()
                    )
                )

            except Exception:

                continue

            if (
                path_a in image_set
                and path_b in image_set
            ):

                near_uf.union(
                    path_a,
                    path_b
                )

        near_groups = defaultdict(list)

        for image in near_uf.parent:

            root = near_uf.find(image)

            near_groups[root].append(
                image
            )

        for index, members in enumerate(
            near_groups.values(),
            start=1,
        ):

            if len(members) < 2:
                continue

            near_id = stable_group_id(
                "NEAR",
                index,
            )

            for image in members:

                near_lookup[
                    image
                ] = near_id

    # ========================================================
    # STATISTICS
    # ========================================================

    multi_image_groups = [
        members
        for members in groups.values()
        if len(members) > 1
    ]

    exact_groups_count = (
        exact_df["duplicate_group"].nunique()
        if EXACT_DUPLICATE_FILE.exists()
        else 0
    )

    near_pairs_count = (
        len(near_df)
        if NEAR_DUPLICATE_FILE.exists()
        else 0
    )

    grouped_images = sum(
        len(members)
        for members in multi_image_groups
    )

    print("\n" + "=" * 75)
    print("DUPLICATE GROUP RESULTS")
    print("=" * 75)

    print(
        f"Source exact duplicate groups: "
        f"{exact_groups_count:,}"
    )

    print(
        f"Strong near-duplicate pairs: "
        f"{near_pairs_count:,}"
    )

    print(
        f"Final leakage groups: "
        f"{len(groups):,}"
    )

    print(
        f"Groups containing >1 image: "
        f"{len(multi_image_groups):,}"
    )

    print(
        f"Images participating in grouped "
        f"duplicates/near-duplicates: "
        f"{grouped_images:,}"
    )

    if multi_image_groups:

        largest = sorted(
            [
                len(members)
                for members in multi_image_groups
            ],
            reverse=True,
        )[:10]

        print(
            "\nLargest leakage groups:"
        )

        for size in largest:

            print(
                f"  {size:,} images"
            )

    return (
        exact_lookup,
        near_lookup,
        leakage_lookup,
    )
# ============================================================
# LOAD TAXONOMY
# ============================================================

def load_taxonomy():

    print("\n" + "=" * 75)
    print("LOADING TAXONOMY")
    print("=" * 75)

    if not TAXONOMY_FILE.exists():

        raise FileNotFoundError(
            f"Taxonomy file not found: "
            f"{TAXONOMY_FILE}"
        )

    df = pd.read_csv(
        TAXONOMY_FILE
    )

    required = {
        "dataset",
        "original_class",
        "unified_parent",
        "unified_class",
        "training_role",
        "safety_level",
        "disposal_group",
        "mapping_confidence",
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            "Taxonomy mapping is missing columns: "
            + ", ".join(sorted(missing))
        )

    print(
        f"Taxonomy rows: {len(df):,}"
    )

    return df


# ============================================================
# TAXONOMY LOOKUP
# ============================================================

def build_taxonomy_lookup(taxonomy_df):

    lookup = {}

    for _, row in taxonomy_df.iterrows():

        key = (
            str(row["dataset"])
            .strip()
            .lower(),

            str(row["original_class"])
            .strip()
            .lower(),
        )

        lookup[key] = row.to_dict()

    return lookup


# ============================================================
# DATASET IDENTIFICATION
# ============================================================

def identify_dataset(path):

    parts = [
        p.lower()
        for p in path.parts
    ]

    for dataset in [
        "trashnet",
        "realwaste",
        "phenomsg",
        "ewaste",
    ]:

        if dataset in parts:

            return dataset

    return "unknown"


# ============================================================
# ORIGINAL CLASS IDENTIFICATION
# ============================================================

def identify_original_class(
    path,
    dataset,
):

    parts = list(path.parts)

    lower_parts = [
        p.lower()
        for p in parts
    ]

    try:

        index = lower_parts.index(
            dataset
        )

    except ValueError:

        return "UNKNOWN"

    remaining = parts[index + 1:]

    if not remaining:

        return "UNKNOWN"

    # E-Waste has train/test/class
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

    return remaining[0]


# ============================================================
# GROUP ASSIGNMENT
# ============================================================

def build_leakage_group(
    image,
    exact_lookup,
    near_lookup,
):

    exact_group = exact_lookup.get(
        image
    )

    near_group = near_lookup.get(
        image
    )

    # --------------------------------------------------------
    # Priority:
    # exact group > near group
    # --------------------------------------------------------

    if exact_group:

        return exact_group

    if near_group:

        return near_group

    # Unique image
    return f"UNIQUE_{hashlib.md5(image.encode()).hexdigest()[:12]}"


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

def assign_group_splits(
    records
):

    print("\n" + "=" * 75)
    print("CREATING GROUP-AWARE SPLITS")
    print("=" * 75)

    random.seed(
        RANDOM_SEED
    )

    groups = defaultdict(list)

    for record in records:

        groups[
            record["leakage_group"]
        ].append(record)

    print(
        f"\nTotal leakage groups: "
        f"{len(groups):,}"
    )

    # --------------------------------------------------------
    # Sort largest groups first
    # --------------------------------------------------------

    group_items = list(
        groups.items()
    )

    random.shuffle(
        group_items
    )

    group_items.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    # --------------------------------------------------------
    # Target counts
    # --------------------------------------------------------

    total_images = len(records)

    target_train = (
        total_images * TRAIN_RATIO
    )

    target_val = (
        total_images * VAL_RATIO
    )

    target_test = (
        total_images * TEST_RATIO
    )

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    group_split = {}

    # --------------------------------------------------------
    # Greedy group assignment
    # --------------------------------------------------------

    for group_id, members in group_items:

        group_size = len(members)

        remaining = {
            "train":
                target_train
                - split_counts["train"],

            "val":
                target_val
                - split_counts["val"],

            "test":
                target_test
                - split_counts["test"],
        }

        # Choose split with greatest remaining capacity
        selected_split = max(
            remaining,
            key=remaining.get,
        )

        group_split[
            group_id
        ] = selected_split

        split_counts[
            selected_split
        ] += group_size

    # --------------------------------------------------------
    # Add split to records
    # --------------------------------------------------------

    for record in records:

        record["split"] = group_split[
            record["leakage_group"]
        ]

    print(
        "\nSplit counts:"
    )

    for split, count in split_counts.items():

        percentage = (
            count / total_images * 100
        )

        print(
            f"  {split:<6}"
            f"{count:>7,}"
            f" ({percentage:6.2f}%)"
        )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("WASTE SEGREGATION AI")
    print("DATASET V1 MANIFEST BUILDER")
    print("=" * 75)

    print(
        "\nIMPORTANT:"
        "\nThis script does NOT modify images."
        "\nIt does NOT delete images."
        "\nIt does NOT move images."
        "\nIt creates metadata/manifests only."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    taxonomy_df = load_taxonomy()

    taxonomy_lookup = build_taxonomy_lookup(
        taxonomy_df
    )

    images = find_all_images()

    print(
        f"\nImages discovered: "
        f"{len(images):,}"
    )

    if len(images) == 0:

        raise RuntimeError(
            "No images found under data/raw."
        )

    # --------------------------------------------------------
    # Duplicate groups
    # --------------------------------------------------------

    (
    exact_lookup,
    near_lookup,
    leakage_lookup,
    ) = build_duplicate_groups(
    images
)

    # --------------------------------------------------------
    # Build records
    # --------------------------------------------------------

    records = []

    unmapped = []

    for path in images:

        relative_path = relative_to_raw(
            path
        )

        dataset = identify_dataset(
            path
        )

        original_class = identify_original_class(
            path,
            dataset,
        )

        taxonomy_key = (
            dataset.lower(),
            original_class.lower(),
        )

        mapping = taxonomy_lookup.get(
            taxonomy_key
        )

        if mapping is None:

            unmapped.append({
                "path": relative_path,
                "dataset": dataset,
                "original_class": original_class,
            })

            continue

        leakage_group = leakage_lookup.get(
    relative_path,
    f"UNIQUE_{hashlib.md5(relative_path.encode()).hexdigest()[:12]}"
)
        

        records.append({

            "image_path":
                relative_path,

            "dataset":
                dataset,

            "original_class":
                original_class,

            "unified_parent":
                mapping["unified_parent"],

            "unified_class":
                mapping["unified_class"],

            "training_role":
                mapping["training_role"],

            "safety_level":
                mapping["safety_level"],

            "disposal_group":
                mapping["disposal_group"],

            "mapping_confidence":
                mapping["mapping_confidence"],

            "leakage_group":
                leakage_group,

            "exact_duplicate_group":
                exact_lookup.get(
                    relative_path,
                    "",
                ),

            "near_duplicate_group":
                near_lookup.get(
                    relative_path,
                    "",
                ),
        })

    # --------------------------------------------------------
    # Mapping validation
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("MAPPING VALIDATION")
    print("=" * 75)

    print(
        f"Mapped images: "
        f"{len(records):,}"
    )

    print(
        f"Unmapped images: "
        f"{len(unmapped):,}"
    )

    if unmapped:

        print(
            "\nWARNING: Unmapped images detected."
        )

        for item in unmapped[:20]:

            print(
                f"  {item['dataset']} / "
                f"{item['original_class']} / "
                f"{item['path']}"
            )

        print(
            "\nOnly the first 20 are shown."
        )

        raise RuntimeError(
            "Dataset V1 cannot be created until "
            "all images have valid taxonomy mappings."
        )

    # --------------------------------------------------------
    # Assign splits
    # --------------------------------------------------------

    records = assign_group_splits(
        records
    )

    manifest_df = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Duplicate group report
    # --------------------------------------------------------

    group_rows = []

    for group_id, group_df in manifest_df.groupby(
        "leakage_group"
    ):

        group_rows.append({

            "leakage_group":
                group_id,

            "image_count":
                len(group_df),

            "split":
                group_df["split"].iloc[0],

            "datasets":
                "|".join(
                    sorted(
                        group_df["dataset"]
                        .unique()
                    )
                ),

            "unified_classes":
                "|".join(
                    sorted(
                        group_df["unified_class"]
                        .unique()
                    )
                ),

            "has_exact_duplicate":
                bool(
                    group_df[
                        "exact_duplicate_group"
                    ].astype(bool).any()
                ),

            "has_near_duplicate":
                bool(
                    group_df[
                        "near_duplicate_group"
                    ].astype(bool).any()
                ),
        })

    groups_df = pd.DataFrame(
        group_rows
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        manifest_df
        .groupby(
            [
                "split",
                "unified_parent",
                "unified_class",
            ],
            as_index=False,
        )
        .agg(
            image_count=(
                "image_path",
                "count",
            )
        )
    )

    # --------------------------------------------------------
    # Write files
    # --------------------------------------------------------

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_df.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8",
    )

    groups_df.to_csv(
        OUTPUT_GROUPS,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 75)
    print("DATASET V1 CREATED")
    print("=" * 75)

    print(
        f"\nManifest:"
        f"\n  {OUTPUT_MANIFEST}"
    )

    print(
        f"\nSummary:"
        f"\n  {OUTPUT_SUMMARY}"
    )

    print(
        f"\nDuplicate groups:"
        f"\n  {OUTPUT_GROUPS}"
    )

    print("\n" + "=" * 75)
    print("CLASS DISTRIBUTION")
    print("=" * 75)

    class_summary = (
        manifest_df
        .groupby(
            "unified_class"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    for class_name, count in class_summary.items():

        print(
            f"{class_name:<35}"
            f"{count:>7,}"
        )

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)

    print(
        "\nNo raw images were modified."
    )


if __name__ == "__main__":
    main()