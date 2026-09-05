from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/reports/taxonomy_source_counts.csv"
)

OUTPUT_FILE = Path(
    "data/reports/taxonomy_mapping.csv"
)


# ============================================================
# TAXONOMY RULES
# ============================================================

MAPPING = {

    # --------------------------------------------------------
    # TRASHNET
    # --------------------------------------------------------

    ("trashnet", "cardboard"): {
        "unified_parent": "Recyclable",
        "unified_class": "Cardboard",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Cardboard Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("trashnet", "glass"): {
        "unified_parent": "Recyclable",
        "unified_class": "Glass",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Glass Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("trashnet", "metal"): {
        "unified_parent": "Recyclable",
        "unified_class": "Metal",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Metal Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("trashnet", "paper"): {
        "unified_parent": "Recyclable",
        "unified_class": "Paper",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Paper Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("trashnet", "plastic"): {
        "unified_parent": "Recyclable",
        "unified_class": "Plastic",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Plastic Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("trashnet", "trash"): {
        "unified_parent": "Residual",
        "unified_class": "Residual",
        "training_role": "residual",
        "safety_level": "S0",
        "disposal_group": "Residual Waste",
        "mapping_confidence": "MEDIUM",
        "mapping_reason": "Generic TrashNet label; safest unified representation is residual waste."
    },


    # --------------------------------------------------------
    # REALWASTE
    # --------------------------------------------------------

    ("realwaste", "cardboard"): {
        "unified_parent": "Recyclable",
        "unified_class": "Cardboard",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Cardboard Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("realwaste", "food organics"): {
        "unified_parent": "Organic",
        "unified_class": "Food Organics",
        "training_role": "organic",
        "safety_level": "S0",
        "disposal_group": "Organic Waste",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct organic-waste label."
    },

    ("realwaste", "glass"): {
        "unified_parent": "Recyclable",
        "unified_class": "Glass",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Glass Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("realwaste", "metal"): {
        "unified_parent": "Recyclable",
        "unified_class": "Metal",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Metal Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("realwaste", "miscellaneous trash"): {
        "unified_parent": "Residual",
        "unified_class": "Residual",
        "training_role": "residual",
        "safety_level": "S0",
        "disposal_group": "Residual Waste",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Dataset explicitly identifies miscellaneous trash."
    },

    ("realwaste", "paper"): {
        "unified_parent": "Recyclable",
        "unified_class": "Paper",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Paper Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("realwaste", "plastic"): {
        "unified_parent": "Recyclable",
        "unified_class": "Plastic",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Plastic Recycling",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct material-level label."
    },

    ("realwaste", "textile trash"): {
        "unified_parent": "Recyclable",
        "unified_class": "Textile",
        "training_role": "material",
        "safety_level": "S1",
        "disposal_group": "Textile Recovery",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct textile-waste label."
    },

    ("realwaste", "vegetation"): {
        "unified_parent": "Organic",
        "unified_class": "Vegetation",
        "training_role": "organic",
        "safety_level": "S0",
        "disposal_group": "Organic Waste",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Direct vegetation/organic label."
    },


    # --------------------------------------------------------
    # E-WASTE
    # --------------------------------------------------------

    ("ewaste", "battery"): {
        "unified_parent": "Special",
        "unified_class": "Battery",
        "training_role": "safety_critical",
        "safety_level": "S3",
        "disposal_group": "Battery Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Battery is explicitly identifiable and requires special handling."
    },

    ("ewaste", "keyboard"): {
        "unified_parent": "E-Waste",
        "unified_class": "Electronic Device",
        "training_role": "e_waste",
        "safety_level": "S2",
        "disposal_group": "E-Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Electronic peripheral."
    },

    ("ewaste", "mobile"): {
        "unified_parent": "E-Waste",
        "unified_class": "Electronic Device",
        "training_role": "e_waste",
        "safety_level": "S2",
        "disposal_group": "E-Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Electronic consumer device."
    },

    ("ewaste", "mouse"): {
        "unified_parent": "E-Waste",
        "unified_class": "Electronic Device",
        "training_role": "e_waste",
        "safety_level": "S2",
        "disposal_group": "E-Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Electronic peripheral."
    },

    ("ewaste", "printer"): {
        "unified_parent": "E-Waste",
        "unified_class": "Electronic Device",
        "training_role": "e_waste",
        "safety_level": "S2",
        "disposal_group": "E-Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Electronic device."
    },

    ("ewaste", "player"): {
        "unified_parent": "E-Waste",
        "unified_class": "Electronic Device",
        "training_role": "e_waste",
        "safety_level": "S2",
        "disposal_group": "E-Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Electronic media device."
    },

    ("ewaste", "pcb"): {
        "unified_parent": "E-Waste",
        "unified_class": "Electronic Component",
        "training_role": "e_waste",
        "safety_level": "S2",
        "disposal_group": "E-Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Printed circuit board/electronic component."
    },

    ("ewaste", "microwave"): {
        "unified_parent": "E-Waste",
        "unified_class": "Large Electronic Appliance",
        "training_role": "e_waste",
        "safety_level": "S2",
        "disposal_group": "E-Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Electrical appliance."
    },

    ("ewaste", "television"): {
        "unified_parent": "E-Waste",
        "unified_class": "Large Electronic Appliance",
        "training_role": "e_waste",
        "safety_level": "S2",
        "disposal_group": "E-Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Electrical/electronic appliance."
    },

    ("ewaste", "washing machine"): {
        "unified_parent": "E-Waste",
        "unified_class": "Large Electronic Appliance",
        "training_role": "e_waste",
        "safety_level": "S2",
        "disposal_group": "E-Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Large electrical appliance."
    },


    # --------------------------------------------------------
    # PHENOMSG
    #
    # IMPORTANT:
    # These are broad source labels.
    # We do NOT pretend they are specific materials.
    # --------------------------------------------------------

    ("phenomsg", "hazardous"): {
        "unified_parent": "Hazardous",
        "unified_class": "Hazardous Waste",
        "training_role": "stream",
        "safety_level": "S3",
        "disposal_group": "Hazardous Waste Collection",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Source dataset explicitly identifies hazardous waste."
    },

    ("phenomsg", "non-recyclable"): {
        "unified_parent": "Residual",
        "unified_class": "Residual",
        "training_role": "stream",
        "safety_level": "S0",
        "disposal_group": "Residual Waste",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Source dataset explicitly identifies non-recyclable waste."
    },

    ("phenomsg", "organic"): {
        "unified_parent": "Organic",
        "unified_class": "Organic Stream",
        "training_role": "stream",
        "safety_level": "S0",
        "disposal_group": "Organic Waste",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Source label identifies the broad organic waste stream without specifying a material."
    },

    ("phenomsg", "recyclable"): {
        "unified_parent": "Recyclable",
        "unified_class": "Recyclable Stream",
        "training_role": "stream",
        "safety_level": "S1",
        "disposal_group": "Recycling Stream",
        "mapping_confidence": "HIGH",
        "mapping_reason": "Source label identifies recyclability but does not specify a material."
    },
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("WASTE SEGREGATION AI")
    print("TAXONOMY MAPPING BUILDER")
    print("=" * 75)

    if not INPUT_FILE.exists():

        print("\nERROR:")
        print(f"Input file not found:")
        print(INPUT_FILE.resolve())
        return

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "dataset",
        "original_class",
        "image_count",
    }

    missing = required_columns - set(df.columns)

    if missing:

        print(
            "\nERROR: Missing columns:"
        )

        for column in missing:
            print(f"  - {column}")

        return

    rows = []

    unmapped = []

    for _, source_row in df.iterrows():

        dataset = str(
            source_row["dataset"]
        ).strip().lower()

        original_class = str(
            source_row["original_class"]
        ).strip()

        key = (
            dataset,
            original_class.lower(),
        )

        source_count = int(
            source_row["image_count"]
        )

        if key in MAPPING:

            mapping = MAPPING[key]

            row = {
                "dataset": dataset,
                "original_class": original_class,
                "source_count": source_count,
                **mapping,
            }

            rows.append(row)

        else:

            unmapped.append({
                "dataset": dataset,
                "original_class": original_class,
                "source_count": source_count,
            })

    # --------------------------------------------------------
    # Write mapping
    # --------------------------------------------------------

    output_df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # REPORT
    # ========================================================

    print("\nMapped classes:")
    print(
        f"  {len(output_df)}"
    )

    print("\nMapped images:")
    print(
        f"  {output_df['source_count'].sum():,}"
    )

    print("\n" + "=" * 75)
    print("UNIFIED CLASS DISTRIBUTION")
    print("=" * 75)

    class_summary = (
        output_df
        .groupby(
            [
                "unified_parent",
                "unified_class",
            ],
            as_index=False,
        )
        ["source_count"]
        .sum()
        .sort_values(
            "source_count",
            ascending=False,
        )
    )

    for _, row in class_summary.iterrows():

        print(
            f"{row['unified_parent']:<15}"
            f"{row['unified_class']:<30}"
            f"{row['source_count']:>7,}"
        )

    # --------------------------------------------------------
    # Unmapped classes
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("UNMAPPED CLASSES")
    print("=" * 75)

    if unmapped:

        for item in unmapped:

            print(
                f"  {item['dataset']:<12}"
                f"{item['original_class']:<30}"
                f"{item['source_count']:>7,}"
            )

        print(
            "\nWARNING:"
            "\nSome source classes were not mapped."
        )

    else:

        print(
            "  ✓ All source classes mapped."
        )

    print("\n" + "=" * 75)
    print("OUTPUT")
    print("=" * 75)

    print(
        f"✓ {OUTPUT_FILE}"
    )

    print(
        "\nNo images were modified."
    )

    print(
        "No images were moved."
    )

    print(
        "No images were deleted."
    )


if __name__ == "__main__":
    main()