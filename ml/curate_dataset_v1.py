from pathlib import Path
import pandas as pd


# ============================================================
# WASTE SEGREGATION AI
# DATASET V1 CURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data" / "reports" / "dataset_v1_manifest.csv"
DECISIONS = ROOT / "data" / "reports" / "conflict_decisions_completed.csv"

OUTPUT_MANIFEST = ROOT / "data" / "reports" / "dataset_v1_curated_manifest.csv"
OUTPUT_AUDIT = ROOT / "data" / "reports" / "dataset_v1_curation_audit.csv"


def main():

    print("=" * 75)
    print("WASTE SEGREGATION AI")
    print("DATASET V1 CURATION")
    print("=" * 75)

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST}"
        )

    if not DECISIONS.exists():
        raise FileNotFoundError(
            f"Conflict decisions not found:\n{DECISIONS}"
        )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    manifest = pd.read_csv(MANIFEST)
    decisions = pd.read_csv(DECISIONS)

    print(f"\nOriginal manifest rows: {len(manifest):,}")
    print(f"Conflict decisions:     {len(decisions):,}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_manifest = {
        "image_path",
        "leakage_group",
        "unified_class",
        "split"
    }

    required_decisions = {
        "leakage_group",
        "decision",
        "reason",
        "action",
        "keep_class",
        "exclude_class"
    }

    missing_manifest = required_manifest - set(manifest.columns)
    missing_decisions = required_decisions - set(decisions.columns)

    if missing_manifest:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing_manifest)}"
        )

    if missing_decisions:
        raise ValueError(
            f"Decision file missing columns: {sorted(missing_decisions)}"
        )

    # --------------------------------------------------------
    # Normalize text fields
    # --------------------------------------------------------

    for col in ["leakage_group", "unified_class", "split"]:
        manifest[col] = manifest[col].astype(str).str.strip()

    for col in [
        "leakage_group",
        "decision",
        "reason",
        "action",
        "keep_class",
        "exclude_class"
    ]:
        decisions[col] = decisions[col].fillna("").astype(str).str.strip()

    # --------------------------------------------------------
    # Validate decisions
    # --------------------------------------------------------

    valid_decisions = {
        "ANNOTATION_ERROR",
        "LEGITIMATE_DIFFERENCE",
        "AMBIGUOUS",
        "NEEDS_REVIEW"
    }

    invalid = decisions[
        ~decisions["decision"].isin(valid_decisions)
    ]

    if len(invalid) > 0:
        print("\nERROR: Invalid decisions found:")
        print(invalid.to_string(index=False))
        raise ValueError("Invalid decision value found.")

    # --------------------------------------------------------
    # Make sure every conflict group appears only once
    # --------------------------------------------------------

    duplicate_groups = decisions[
        decisions["leakage_group"].duplicated(keep=False)
    ]

    if len(duplicate_groups) > 0:
        print("\nERROR: Duplicate leakage groups in decision file:")
        print(duplicate_groups.to_string(index=False))
        raise ValueError("Duplicate leakage groups found.")

    # --------------------------------------------------------
    # Start with everything included
    # --------------------------------------------------------

    curated = manifest.copy()

    curated["curation_status"] = "KEEP"
    curated["curation_reason"] = ""
    curated["curation_action"] = ""

    # --------------------------------------------------------
    # Apply decisions
    # --------------------------------------------------------

    audit_rows = []

    for _, decision in decisions.iterrows():

        group = decision["leakage_group"]
        decision_type = decision["decision"]
        reason = decision["reason"]
        action = decision["action"]
        keep_class = decision["keep_class"]
        exclude_class = decision["exclude_class"]

        group_mask = curated["leakage_group"] == group

        group_count = group_mask.sum()

        if group_count == 0:
            print(
                f"\nWARNING: {group} not found in manifest."
            )
            continue

        # ----------------------------------------------------
        # Legitimate difference
        # ----------------------------------------------------

        if decision_type == "LEGITIMATE_DIFFERENCE":

            curated.loc[group_mask, "curation_status"] = "KEEP"
            curated.loc[group_mask, "curation_reason"] = reason
            curated.loc[group_mask, "curation_action"] = "KEEP"

        # ----------------------------------------------------
        # Annotation error
        # ----------------------------------------------------

        elif decision_type == "ANNOTATION_ERROR":

            if not exclude_class:
                raise ValueError(
                    f"{group}: exclude_class is empty."
                )

            if not keep_class:
                raise ValueError(
                    f"{group}: keep_class is empty."
                )

            # Make sure the expected conflicting classes exist
            group_classes = set(
                curated.loc[group_mask, "unified_class"]
            )

            if exclude_class not in group_classes:
                raise ValueError(
                    f"{group}: exclude_class "
                    f"'{exclude_class}' not found. "
                    f"Available classes: {sorted(group_classes)}"
                )

            # Exclude only the incorrectly labelled class
            exclude_mask = (
                group_mask
                & (
                    curated["unified_class"]
                    == exclude_class
                )
            )

            curated.loc[
                exclude_mask,
                "curation_status"
            ] = "EXCLUDE"

            curated.loc[
                exclude_mask,
                "curation_reason"
            ] = reason

            curated.loc[
                exclude_mask,
                "curation_action"
            ] = action

            # Keep the correct class
            keep_mask = (
                group_mask
                & (
                    curated["unified_class"]
                    == keep_class
                )
            )

            curated.loc[
                keep_mask,
                "curation_status"
            ] = "KEEP"

            curated.loc[
                keep_mask,
                "curation_reason"
            ] = reason

            curated.loc[
                keep_mask,
                "curation_action"
            ] = "KEEP_CORRECT_LABEL"

        # ----------------------------------------------------
        # Ambiguous
        # ----------------------------------------------------

        elif decision_type == "AMBIGUOUS":

            curated.loc[
                group_mask,
                "curation_status"
            ] = "KEEP_HARD_CASE"

            curated.loc[
                group_mask,
                "curation_reason"
            ] = reason

            curated.loc[
                group_mask,
                "curation_action"
            ] = "KEEP_HARD_CASE"

        # ----------------------------------------------------
        # Needs review
        # ----------------------------------------------------

        elif decision_type == "NEEDS_REVIEW":

            curated.loc[
                group_mask,
                "curation_status"
            ] = "HOLD"

            curated.loc[
                group_mask,
                "curation_reason"
            ] = reason

            curated.loc[
                group_mask,
                "curation_action"
            ] = "HOLD"

        # ----------------------------------------------------
        # Audit record
        # ----------------------------------------------------

        audit_rows.append({
            "leakage_group": group,
            "decision": decision_type,
            "reason": reason,
            "action": action,
            "keep_class": keep_class,
            "exclude_class": exclude_class,
            "images_in_group": group_count
        })

    # --------------------------------------------------------
    # Save curated manifest
    # --------------------------------------------------------

    curated.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    audit = pd.DataFrame(audit_rows)

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("CURATION RESULT")
    print("=" * 75)

    print(
        f"\nOriginal images:       {len(manifest):,}"
    )

    print(
        "Excluded images:      "
        f"{(curated['curation_status'] == 'EXCLUDE').sum():,}"
    )

    print(
        "Kept images:          "
        f"{(curated['curation_status'] != 'EXCLUDE').sum():,}"
    )

    print("\nCuration status:")

    print(
        curated["curation_status"]
        .value_counts()
        .to_string()
    )

    print("\nDecision distribution:")

    print(
        decisions["decision"]
        .value_counts()
        .to_string()
    )

    print("\nCurated class distribution:")

    kept = curated[
        curated["curation_status"] != "EXCLUDE"
    ]

    print(
        kept["unified_class"]
        .value_counts()
        .to_string()
    )

    print("\nSplit distribution:")

    print(
        kept["split"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("OUTPUT")
    print("=" * 75)

    print(f"\n✓ {OUTPUT_MANIFEST}")
    print(f"✓ {OUTPUT_AUDIT}")

    print("\nIMPORTANT:")
    print("✓ data/raw was NOT modified.")
    print("✓ No images were deleted.")
    print("✓ No images were moved.")
    print("✓ No images were renamed.")


if __name__ == "__main__":
    main()