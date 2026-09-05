from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "dataset_v1_manifest.csv"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "label_conflict_groups.csv"
)


def main():

    print("=" * 75)
    print("WASTE SEGREGATION AI")
    print("LABEL CONFLICT AUDIT")
    print("=" * 75)

    df = pd.read_csv(MANIFEST)

    grouped = (
        df.groupby("leakage_group")
        .agg(
            image_count=("image_path", "count"),
            class_count=("unified_class", "nunique"),
            classes=(
                "unified_class",
                lambda s: " | ".join(sorted(set(s)))
            ),
            images=(
                "image_path",
                lambda s: " | ".join(s)
            ),
        )
        .reset_index()
    )

    conflicts = grouped[
        grouped["class_count"] > 1
    ].copy()

    conflicts = conflicts.sort_values(
        ["class_count", "image_count"],
        ascending=[False, False]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conflicts.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("Total leakage groups:", len(grouped))
    print("Conflicting groups:", len(conflicts))

    print()
    print("Conflict distribution:")
    print(
        conflicts["class_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("OUTPUT:")
    print(OUTPUT)

    print()
    print("Top conflicts:")
    print(
        conflicts.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()