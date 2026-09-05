from pathlib import Path
import math
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = PROJECT_ROOT / "data/reports/dataset_v1_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/reports/conflict_review"

IMAGE_SIZE = (220, 180)
TEXT_HEIGHT = 100
COLUMNS = 3


def get_font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


def make_group_sheet(group_id, group_df):
    items = []

    for _, row in group_df.iterrows():
        image_path = PROJECT_ROOT / "data/raw" / row["image_path"]

        if not image_path.exists():
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail(IMAGE_SIZE)

            canvas = Image.new(
                "RGB",
                IMAGE_SIZE,
                "white"
            )

            x = (IMAGE_SIZE[0] - img.width) // 2
            y = (IMAGE_SIZE[1] - img.height) // 2

            canvas.paste(img, (x, y))

            items.append({
                "image": canvas,
                "source": row["dataset"],
                "class": row["unified_class"],
                "file": Path(row["image_path"]).name,
            })

        except Exception as e:
            print(f"Could not open: {image_path}")
            print(e)

    if not items:
        return

    rows = math.ceil(len(items) / COLUMNS)

    sheet_width = COLUMNS * IMAGE_SIZE[0]
    sheet_height = rows * (IMAGE_SIZE[1] + TEXT_HEIGHT)

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    font = get_font(14)
    small_font = get_font(11)

    for i, item in enumerate(items):

        col = i % COLUMNS
        row = i // COLUMNS

        x = col * IMAGE_SIZE[0]
        y = row * (IMAGE_SIZE[1] + TEXT_HEIGHT)

        sheet.paste(
            item["image"],
            (x, y)
        )

        draw.text(
            (x + 5, y + IMAGE_SIZE[1] + 5),
            f"Source: {item['source']}",
            fill="black",
            font=font
        )

        draw.text(
            (x + 5, y + IMAGE_SIZE[1] + 25),
            f"Class: {item['class']}",
            fill="black",
            font=font
        )

        filename = item["file"]

        if len(filename) > 32:
            filename = filename[:29] + "..."

        draw.text(
            (x + 5, y + IMAGE_SIZE[1] + 47),
            filename,
            fill="black",
            font=small_font
        )

    output_file = OUTPUT_DIR / f"{group_id}.jpg"
    sheet.save(output_file, quality=95)

    print(f"Created: {output_file}")


def main():

    print("=" * 75)
    print("WASTE SEGREGATION AI")
    print("CONFLICT VISUAL REVIEW")
    print("=" * 75)

    df = pd.read_csv(MANIFEST)

    group_counts = df.groupby("leakage_group").size()

    conflicting_groups = group_counts[group_counts > 1].index

    # Only groups containing multiple classes
    conflicts = []

    for group_id in conflicting_groups:

        group = df[df["leakage_group"] == group_id]

        if group["unified_class"].nunique() > 1:
            conflicts.append(group_id)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Conflicting groups:", len(conflicts))
    print("Output directory:", OUTPUT_DIR)
    print()

    for group_id in conflicts:

        group_df = df[
            df["leakage_group"] == group_id
        ]

        make_group_sheet(
            group_id,
            group_df
        )

    print()
    print("=" * 75)
    print("REVIEW FILES CREATED")
    print("=" * 75)


if __name__ == "__main__":
    main()