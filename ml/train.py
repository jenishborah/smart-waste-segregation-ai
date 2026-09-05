"""
WASTE SEGREGATION AI
MODEL TRAINING - DATASET V1

Input:
    data/reports/dataset_v1_curated_manifest.csv

Images:
    data/raw/<image_path>

Output:
    models/
        best_model.pth
        last_model.pth
        class_names.json
        training_history.csv
"""

from pathlib import Path
import json
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "dataset_v1_curated_manifest.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 0.0003

NUM_WORKERS = 0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("WASTE SEGREGATION AI")
print("MODEL TRAINING")
print("=" * 75)

print()
print("Device:", DEVICE)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if DEVICE.type == "cpu":
    print()
    print("WARNING: CUDA is not available.")
    print("Training will run on CPU.")
    print("This is acceptable for the first training run.")
else:
    print("GPU:", torch.cuda.get_device_name(0))

print()


# ============================================================
# DATASET CLASS
# ============================================================

class WasteDataset(Dataset):

    def __init__(
        self,
        dataframe,
        class_to_idx,
        transform=None
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.class_to_idx = class_to_idx

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # image_path is relative to data/raw
        image_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / str(row["image_path"])
        )

        class_name = str(
            row["unified_class"]
        )

        label = self.class_to_idx[
            class_name
        ]

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"\nCould not load image:\n"
                f"{image_path}\n"
                f"Error: {e}"
            )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# LOAD MANIFEST
# ============================================================

print("=" * 75)
print("LOADING DATASET")
print("=" * 75)

if not MANIFEST.exists():

    raise FileNotFoundError(
        f"\nManifest not found:\n{MANIFEST}"
    )


df = pd.read_csv(
    MANIFEST
)

print()
print("Manifest:", MANIFEST)
print("Total images:", len(df))


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "image_path",
    "unified_class",
    "split",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Manifest is missing columns: "
        + str(missing_columns)
    )

print("Required columns: OK")


# ============================================================
# IMAGE EXISTENCE CHECK
# ============================================================

print()
print("Checking image paths...")

missing_images = []

for image_path in df["image_path"]:

    full_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / str(image_path)
    )

    if not full_path.exists():

        missing_images.append(
            str(image_path)
        )


if missing_images:

    print()
    print(
        f"ERROR: {len(missing_images)} "
        "images are missing."
    )

    for path in missing_images[:20]:

        print("  ", path)

    raise FileNotFoundError(
        "Dataset contains missing images."
    )


print(
    f"✓ All {len(df):,} images exist"
)


# ============================================================
# CLASSES
# ============================================================

classes = sorted(
    df[
        "unified_class"
    ].dropna().unique().tolist()
)

NUM_CLASSES = len(classes)

class_to_idx = {
    class_name: index
    for index, class_name in enumerate(
        classes
    )
}

idx_to_class = {
    index: class_name
    for class_name, index
    in class_to_idx.items()
}


print()
print("=" * 75)
print("CLASSES")
print("=" * 75)

print()
print("Number of classes:", NUM_CLASSES)

for index, class_name in enumerate(classes):

    count = (
        df["unified_class"]
        == class_name
    ).sum()

    print(
        f"  {index:2d}  "
        f"{class_name:<30} "
        f"{count:>5}"
    )


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

class_file = (
    MODEL_DIR
    / "class_names.json"
)

with open(
    class_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "classes": classes,

            "class_to_idx":
                class_to_idx,

            "idx_to_class": {
                str(index): name
                for index, name
                in idx_to_class.items()
            },
        },
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SPLIT DATA
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print()
print("=" * 75)
print("DATA SPLITS")
print("=" * 75)

print()
print(
    f"Train:      {len(train_df):,}"
)

print(
    f"Validation: {len(val_df):,}"
)

print(
    f"Test:       {len(test_df):,}"
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


validation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = WasteDataset(
    train_df,
    class_to_idx,
    train_transform
)

val_dataset = WasteDataset(
    val_df,
    class_to_idx,
    validation_transform
)

test_dataset = WasteDataset(
    test_df,
    class_to_idx,
    validation_transform
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 75)
print("BUILDING MODEL")
print("=" * 75)

print()
print("Architecture: EfficientNet-B0")
print("Transfer learning: YES")
print("Image size:", IMAGE_SIZE)
print("Output classes:", NUM_CLASSES)


weights = (
    models.EfficientNet_B0_Weights.DEFAULT
)

model = models.efficientnet_b0(
    weights=weights
)


# Replace final classifier

in_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for batch_index, (
        images,
        labels
    ) in enumerate(train_loader):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = (
            outputs.argmax(dim=1)
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        if (
            batch_index + 1
        ) % 50 == 0:

            print(
                f"    Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"
            )


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = (
                outputs.argmax(dim=1)
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    loss = (
        running_loss / total
    )

    accuracy = (
        correct / total
    )


    return loss, accuracy


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("=" * 75)
print("STARTING TRAINING")
print("=" * 75)

print()
print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print()


history = []

best_val_accuracy = 0.0

best_val_loss = float("inf")


best_model_path = (
    MODEL_DIR
    / "best_model.pth"
)

last_model_path = (
    MODEL_DIR
    / "last_model.pth"
)


training_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()


    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print("-" * 75)


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = (
        evaluate(val_loader)
    )


    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    epoch_time = (
        time.time()
        - epoch_start
    )


    print()

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.7f}"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time:.1f}s"
    )


    history.append({

        "epoch": epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "learning_rate":
            current_lr,

        "epoch_time_seconds":
            epoch_time,
    })


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if (
        val_accuracy
        > best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )

        best_val_loss = (
            val_loss
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_to_idx":
                    class_to_idx,

                "classes":
                    classes,

                "image_size":
                    IMAGE_SIZE,

                "architecture":
                    "efficientnet_b0",

                "val_accuracy":
                    val_accuracy,

                "val_loss":
                    val_loss,

                "epoch":
                    epoch,
            },
            best_model_path
        )


        print()
        print(
            "✓ New best model saved"
        )


# ============================================================
# SAVE LAST MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_to_idx":
            class_to_idx,

        "classes":
            classes,

        "image_size":
            IMAGE_SIZE,

        "architecture":
            "efficientnet_b0",

        "epoch":
            EPOCHS,

    },
    last_model_path
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_path = (
    MODEL_DIR
    / "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

test_loss, test_accuracy = (
    evaluate(test_loader)
)

print()

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

total_time = (
    time.time()
    - training_start
)


print()
print("=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print()

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Training Time: "
    f"{total_time / 60:.1f} minutes"
)

print()
print("OUTPUT FILES:")
print(
    f"  ✓ {best_model_path}"
)
print(
    f"  ✓ {last_model_path}"
)
print(
    f"  ✓ {class_file}"
)
print(
    f"  ✓ {history_path}"
)

print()
print("=" * 75)