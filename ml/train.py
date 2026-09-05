"""
WASTE SEGREGATION AI
Dataset V1 - Image Classification Training

Designed for:
- Google Colab GPU
- Google Drive dataset
- GitHub source code

Dataset:
    data/reports/dataset_v1_curated_manifest.csv

Expected manifest columns:
    image_path
    unified_class
    split

Expected Drive structure:
    DATA_ROOT/
    ├── raw/
    │   ├── trashnet/
    │   ├── realwaste/
    │   ├── ewaste/
    │   └── phenomsg/
    └── reports/
        └── dataset_v1_curated_manifest.csv

The manifest contains paths relative to data/raw.
"""

from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATA_ROOT = "/content/drive/MyDrive/smart-waste-segregation-ai"

MANIFEST_RELATIVE = Path(
    "reports/dataset_v1_curated_manifest.csv"
)

RAW_RELATIVE = Path("raw")

OUTPUT_RELATIVE = Path("training_outputs")

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 2

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 5

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("=" * 70)
        print("GPU ENABLED")
        print("=" * 70)
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    else:
        device = torch.device("cpu")

        print("=" * 70)
        print("WARNING: GPU NOT AVAILABLE")
        print("=" * 70)
        print("Training will run on CPU.")

    print("Device:", device)

    return device


# ============================================================
# DATASET
# ============================================================

class WasteDataset(Dataset):

    def __init__(
        self,
        dataframe,
        raw_root,
        class_to_idx,
        transform=None,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.raw_root = Path(raw_root)

        self.class_to_idx = class_to_idx

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        relative_path = Path(str(row["image_path"]))

        image_path = self.raw_root / relative_path

        try:

            image = Image.open(image_path).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not load image:\n{image_path}\n\n"
                f"Error: {e}"
            )

        label_name = str(row["unified_class"])

        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

def create_transforms():

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
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    eval_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    return train_transform, eval_transform


# ============================================================
# MODEL
# ============================================================

def create_model(num_classes):

    print()
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    # ResNet18 is a good initial baseline:
    # relatively lightweight and suitable for later Android deployment.

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    print("Architecture: ResNet18")
    print("Classes:", num_classes)

    return model


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return (
        epoch_loss,
        epoch_accuracy,
        all_labels,
        all_predictions,
    )


# ============================================================
# TEST
# ============================================================

def test_model(
    model,
    loader,
    device,
    class_names,
    output_dir,
):

    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    model.eval()

    all_labels = []

    all_predictions = []

    all_confidences = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predictions = (
                probabilities.max(dim=1)
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_confidences.extend(
                confidence.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print(
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = (
        output_dir /
        "classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    cm_path = (
        output_dir /
        "confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    results = {
        "test_accuracy": float(accuracy),
        "num_test_images": int(len(all_labels)),
        "classes": class_names,
    }

    with open(
        output_dir / "test_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("Classification report:")
    print(report_df.round(4).to_string())

    print()
    print("Saved:")
    print(report_path)
    print(cm_path)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Google Drive dataset root",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )

    args = parser.parse_args()

    set_seed()

    print()
    print("=" * 70)
    print("WASTE SEGREGATION AI")
    print("MODEL TRAINING")
    print("=" * 70)

    data_root = Path(
        args.data_root
    ).expanduser().resolve()

    manifest_path = (
        data_root /
        MANIFEST_RELATIVE
    )

    raw_root = (
        data_root /
        RAW_RELATIVE
    )

    output_dir = (
        data_root /
        OUTPUT_RELATIVE
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("DATA ROOT:")
    print(data_root)

    print()
    print("MANIFEST:")
    print(manifest_path)

    print()
    print("RAW IMAGE ROOT:")
    print(raw_root)

    print()
    print("OUTPUT:")
    print(output_dir)

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"\nManifest not found:\n"
            f"{manifest_path}\n\n"
            f"Check your Google Drive path."
        )

    if not raw_root.exists():

        raise FileNotFoundError(
            f"\nRaw dataset directory not found:\n"
            f"{raw_root}"
        )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING MANIFEST")
    print("=" * 70)

    df = pd.read_csv(
        manifest_path
    )

    required_columns = [
        "image_path",
        "unified_class",
        "split",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    print("Total images:", len(df))

    print()
    print("Splits:")
    print(
        df["split"].value_counts()
    )

    print()
    print(
        "Classes:",
        df["unified_class"].nunique()
    )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    class_names = sorted(
        df["unified_class"]
        .astype(str)
        .unique()
        .tolist()
    )

    class_to_idx = {
        name: index
        for index, name
        in enumerate(class_names)
    }

    print()
    print("CLASS MAPPING")

    for name, index in class_to_idx.items():

        print(
            f"{index:2d} -> {name}"
        )

    # Save class mapping

    with open(
        output_dir / "class_names.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            class_names,
            f,
            indent=2,
        )

    with open(
        output_dir / "class_to_idx.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            class_to_idx,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

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
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    train_transform, eval_transform = (
        create_transforms()
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = WasteDataset(
        train_df,
        raw_root,
        class_to_idx,
        train_transform,
    )

    val_dataset = WasteDataset(
        val_df,
        raw_root,
        class_to_idx,
        eval_transform,
    )

    test_dataset = WasteDataset(
        test_df,
        raw_root,
        class_to_idx,
        eval_transform,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        len(class_names)
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss / Optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_val_accuracy = 0.0

    epochs_without_improvement = 0

    best_model_path = (
        output_dir /
        "best_model.pth"
    )

    print()
    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(
        1,
        args.epochs + 1
    ):

        epoch_start = time.time()

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        (
            val_loss,
            val_accuracy,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(
            val_accuracy
        )

        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = (
            time.time() -
            epoch_start
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}% | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "learning_rate": float(current_lr),
        })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    class_names,

                "class_to_idx":
                    class_to_idx,

                "image_size":
                    IMAGE_SIZE,

                "architecture":
                    "resnet18",

                "best_val_accuracy":
                    float(best_val_accuracy),
            }

            torch.save(
                checkpoint,
                best_model_path,
            )

            print(
                f"  ✓ New best model saved "
                f"({best_val_accuracy * 100:.2f}%)"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    total_training_time = (
        time.time() -
        start_time
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        output_dir /
        "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Best validation accuracy:",
        f"{checkpoint['best_val_accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_model(
        model,
        test_loader,
        device,
        class_names,
        output_dir,
    )

    # --------------------------------------------------------
    # Training metadata
    # --------------------------------------------------------

    metadata = {
        "architecture": "resnet18",
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "learning_rate": args.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "device": str(device),
        "best_validation_accuracy":
            float(best_val_accuracy),
        "training_time_seconds":
            float(total_training_time),
        "train_images": len(train_df),
        "validation_images": len(val_df),
        "test_images": len(test_df),
    }

    with open(
        output_dir / "training_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best Val Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Training time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print()
    print("OUTPUT FILES:")

    for path in sorted(
        output_dir.iterdir()
    ):

        print(
            " ",
            path.name
        )

    print()
    print(
        "Dataset was NOT modified."
    )


if __name__ == "__main__":
    main()