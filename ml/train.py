"""
WASTE SEGREGATION AI
MODEL TRAINING PIPELINE

Dataset:
    13,196 curated images
    16 unified classes

Manifest:
    data/reports/dataset_v1_curated_manifest.csv

Images:
    data/raw/

Configuration:
    configs/training_config.json

Usage:
    python ml/train.py

Optional:
    python ml/train.py --config configs/training_config.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# CONFIG
# ============================================================

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(project_root, value):
    path = Path(value)

    if path.is_absolute():
        return path

    return project_root / path


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()

        if isinstance(obj, Path):
            return str(obj)

        raise TypeError(
            f"Object of type {type(obj).__name__} "
            "is not JSON serializable"
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            default=convert,
        )


# ============================================================
# DATASET
# ============================================================

class WasteDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_root,
        class_to_idx,
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        relative_path = Path(str(row["image_path"]))

        image_path = self.image_root / relative_path

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as exc:

            raise RuntimeError(
                f"Could not read image:\n{image_path}\n\n{exc}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        class_name = str(row["unified_class"])

        label = self.class_to_idx[class_name]

        return image, label


# ============================================================
# MODEL
# ============================================================

def build_model(
    architecture,
    num_classes,
):

    architecture = architecture.lower().strip()

    if architecture == "resnet18":

        try:
            weights = models.ResNet18_Weights.DEFAULT

            model = models.resnet18(
                weights=weights
            )

        except Exception:

            print(
                "WARNING: Could not load pretrained "
                "ResNet18 weights."
            )

            model = models.resnet18(
                weights=None
            )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

        return model

    if architecture == "resnet50":

        try:
            weights = models.ResNet50_Weights.DEFAULT

            model = models.resnet50(
                weights=weights
            )

        except Exception:

            print(
                "WARNING: Could not load pretrained "
                "ResNet50 weights."
            )

            model = models.resnet50(
                weights=None
            )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

        return model

    raise ValueError(
        f"Unsupported architecture: {architecture}\n"
        "Supported: resnet18, resnet50"
    )


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training,
):

    if training:
        model.train()

    else:
        model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets,
            )

            if training:

                loss.backward()

                optimizer.step()

        predictions = outputs.argmax(
            dim=1
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        all_targets.extend(
            targets.detach()
            .cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

    total = len(all_targets)

    if total == 0:

        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
        }

    return {

        "loss":
            running_loss / total,

        "accuracy":
            accuracy_score(
                all_targets,
                all_predictions,
            ),

        "macro_f1":
            f1_score(
                all_targets,
                all_predictions,
                average="macro",
                zero_division=0,
            ),
    }


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_all(
    model,
    loader,
    device,
):

    model.eval()

    all_targets = []
    all_predictions = []
    all_confidences = []

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidences, predictions = (
            probabilities.max(dim=1)
        )

        all_targets.extend(
            targets.tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_confidences.extend(
            confidences.cpu().tolist()
        )

    return (
        np.array(all_targets),
        np.array(all_predictions),
        np.array(all_confidences),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train waste segregation model"
    )

    parser.add_argument(
        "--config",
        default="configs/training_config.json",
        help="Training configuration file",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # PROJECT PATHS
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    config_path = resolve_path(
        project_root,
        args.config,
    )

    config = load_config(
        config_path
    )

    # --------------------------------------------------------
    # CONFIG VALUES
    # --------------------------------------------------------

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    image_size = int(
        config.get(
            "image_size",
            224,
        )
    )

    batch_size = int(
        config.get(
            "batch_size",
            32,
        )
    )

    epochs = int(
        config.get(
            "epochs",
            20,
        )
    )

    learning_rate = float(
        config.get(
            "learning_rate",
            1e-4,
        )
    )

    weight_decay = float(
        config.get(
            "weight_decay",
            1e-4,
        )
    )

    patience = int(
        config.get(
            "patience",
            5,
        )
    )

    num_workers = int(
        config.get(
            "num_workers",
            2,
        )
    )

    architecture = str(
        config.get(
            "architecture",
            "resnet18",
        )
    )

    set_seed(seed)

    # --------------------------------------------------------
    # DATASET CONFIG
    # --------------------------------------------------------

    dataset_cfg = config.get(
        "dataset",
        {}
    )

    manifest_value = dataset_cfg.get(
        "manifest",
        "data/reports/dataset_v1_curated_manifest.csv",
    )

    manifest_path = resolve_path(
        project_root,
        manifest_value,
    )

    # Support configs using:
    # reports/...
    # instead of:
    # data/reports/...

    if not manifest_path.exists():

        alternative = (
            project_root
            / "data"
            / manifest_value
        )

        if alternative.exists():

            manifest_path = alternative

    raw_value = dataset_cfg.get(
        "raw_directory",
        "data/raw",
    )

    raw_directory = resolve_path(
        project_root,
        raw_value,
    )

    if not raw_directory.exists():

        alternative_raw = (
            project_root
            / "data"
            / raw_value
        )

        if alternative_raw.exists():

            raw_directory = alternative_raw

    # --------------------------------------------------------
    # OUTPUT CONFIG
    # --------------------------------------------------------

    output_cfg = config.get(
        "output",
        {}
    )

    output_directory = output_cfg.get(
        "directory",
        "training_outputs",
    )

    output_dir = resolve_path(
        project_root,
        output_directory,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print("=" * 72)
    print("WASTE SEGREGATION AI")
    print("MODEL TRAINING")
    print("=" * 72)

    print(
        f"Project root:       {project_root}"
    )

    print(
        f"Config:             {config_path}"
    )

    print(
        f"Manifest:           {manifest_path}"
    )

    print(
        f"Image root:         {raw_directory}"
    )

    print(
        f"Output directory:   {output_dir}"
    )

    print(
        f"Architecture:       {architecture}"
    )

    print(
        f"Image size:         {image_size}"
    )

    print(
        f"Batch size:         {batch_size}"
    )

    print(
        f"Epochs:             {epochs}"
    )

    print(
        f"Learning rate:      {learning_rate}"
    )

    print(
        f"Device:             {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU:                "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

    # --------------------------------------------------------
    # VALIDATE PATHS
    # --------------------------------------------------------

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"\nManifest not found:\n"
            f"{manifest_path}\n"
        )

    if not raw_directory.exists():

        raise FileNotFoundError(
            f"\nImage directory not found:\n"
            f"{raw_directory}\n"
        )

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    print("\nLoading dataset manifest...")

    df = pd.read_csv(
        manifest_path
    )

    required_columns = [
        "image_path",
        "unified_class",
        "split",
    ]

    missing_columns = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Manifest is missing columns: "
            f"{missing_columns}"
        )

    df = df.dropna(
        subset=required_columns
    ).copy()

    # --------------------------------------------------------
    # CLASSES
    # --------------------------------------------------------

    classes = sorted(
        df["unified_class"]
        .astype(str)
        .unique()
    )

    num_classes = len(classes)

    class_to_idx = {
        name: index
        for index, name
        in enumerate(classes)
    }

    idx_to_class = {
        str(index): name
        for name, index
        in class_to_idx.items()
    }

    configured_classes = config.get(
        "num_classes"
    )

    if (
        configured_classes is not None
        and int(configured_classes)
        != num_classes
    ):

        raise ValueError(
            f"Config says {configured_classes} "
            f"classes, but manifest contains "
            f"{num_classes}."
        )

    print(
        f"\nTotal images:       {len(df):,}"
    )

    print(
        f"Number of classes:  {num_classes}"
    )

    print("\nClass mapping:")

    for index, name in enumerate(classes):

        print(
            f"  {index:2d} -> {name}"
        )

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    train_df = df[
        df["split"].astype(str)
        == "train"
    ].copy()

    val_df = df[
        df["split"].astype(str)
        == "val"
    ].copy()

    test_df = df[
        df["split"].astype(str)
        == "test"
    ].copy()

    print("\nSplits:")

    print(
        f"  Train:             {len(train_df):,}"
    )

    print(
        f"  Validation:        {len(val_df):,}"
    )

    print(
        f"  Test:              {len(test_df):,}"
    )

    # --------------------------------------------------------
    # IMAGE EXISTENCE
    # --------------------------------------------------------

    print(
        "\nChecking image files..."
    )

    missing_images = []

    for relative_path in (
        df["image_path"]
        .astype(str)
    ):

        image_path = (
            raw_directory
            / Path(relative_path)
        )

        if not image_path.exists():

            missing_images.append(
                str(image_path)
            )

    if missing_images:

        print(
            f"Missing images: "
            f"{len(missing_images)}"
        )

        for path in missing_images[:10]:

            print(
                f"  {path}"
            )

        raise FileNotFoundError(
            "Some images referenced by "
            "the manifest are missing."
        )

    print(
        f"All {len(df):,} image files found."
    )

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    imagenet_mean = [
        0.485,
        0.456,
        0.406,
    ]

    imagenet_std = [
        0.229,
        0.224,
        0.225,
    ]

    train_transform = transforms.Compose([

        transforms.Resize(
            (
                image_size,
                image_size,
            )
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            imagenet_mean,
            imagenet_std,
        ),
    ])

    eval_transform = transforms.Compose([

        transforms.Resize(
            (
                image_size,
                image_size,
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            imagenet_mean,
            imagenet_std,
        ),
    ])

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = WasteDataset(
        train_df,
        raw_directory,
        class_to_idx,
        train_transform,
    )

    val_dataset = WasteDataset(
        val_df,
        raw_directory,
        class_to_idx,
        eval_transform,
    )

    test_dataset = WasteDataset(
        test_df,
        raw_directory,
        class_to_idx,
        eval_transform,
    )

    pin_memory = (
        device.type == "cuda"
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nBuilding model..."
    )

    model = build_model(
        architecture,
        num_classes,
    )

    model = model.to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters:         "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # LOSS / OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    # --------------------------------------------------------
    # SAVE CLASS MAPPING
    # --------------------------------------------------------

    class_names_path = (
        output_dir
        / output_cfg.get(
            "class_names",
            "class_names.json",
        )
    )

    class_to_idx_path = (
        output_dir
        / output_cfg.get(
            "class_to_idx",
            "class_to_idx.json",
        )
    )

    save_json(
        class_names_path,
        {
            "classes": classes,
            "num_classes": num_classes,
        },
    )

    save_json(
        class_to_idx_path,
        {
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        },
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("TRAINING STARTED")
    print("=" * 72)

    history = []

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = (
        output_dir
        / output_cfg.get(
            "best_model",
            "best_model.pth",
        )
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            training=False,
        )

        scheduler.step(
            val_metrics["macro_f1"]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        row = {

            "epoch":
                epoch,

            "learning_rate":
                current_lr,

            "train_loss":
                train_metrics["loss"],

            "train_accuracy":
                train_metrics["accuracy"],

            "train_macro_f1":
                train_metrics["macro_f1"],

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_macro_f1":
                val_metrics["macro_f1"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: "
            f"{train_metrics['loss']:.4f} | "
            f"Train Acc: "
            f"{train_metrics['accuracy']:.4f} | "
            f"Val Loss: "
            f"{val_metrics['loss']:.4f} | "
            f"Val Acc: "
            f"{val_metrics['accuracy']:.4f} | "
            f"Val F1: "
            f"{val_metrics['macro_f1']:.4f} | "
            f"LR: "
            f"{current_lr:.2e}"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if (
            val_metrics["macro_f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch":
                        epoch,

                    "architecture":
                        architecture,

                    "model_state_dict":
                        model.state_dict(),

                    "class_to_idx":
                        class_to_idx,

                    "idx_to_class":
                        idx_to_class,

                    "num_classes":
                        num_classes,

                    "image_size":
                        image_size,

                    "best_val_macro_f1":
                        best_val_f1,

                    "config":
                        config,
                },
                best_model_path,
            )

            print(
                f"  -> Saved best model "
                f"(Val Macro-F1: "
                f"{best_val_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping."
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            break

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        output_dir
        / output_cfg.get(
            "history",
            "training_history.csv",
        )
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print(
        "\nLoading best model..."
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print(
        "\nRunning final test evaluation..."
    )

    (
        test_targets,
        test_predictions,
        test_confidences,
    ) = predict_all(
        model,
        test_loader,
        device,
    )

    test_accuracy = accuracy_score(
        test_targets,
        test_predictions,
    )

    test_macro_f1 = f1_score(
        test_targets,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report_dict = classification_report(
        test_targets,
        test_predictions,
        labels=list(
            range(num_classes)
        ),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report_dict
    )

    classification_report_path = (
        output_dir
        / output_cfg.get(
            "classification_report",
            "classification_report.csv",
        )
    )

    report_df.to_csv(
        classification_report_path
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        test_targets,
        test_predictions,
        labels=list(
            range(num_classes)
        ),
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    )

    confusion_matrix_path = (
        output_dir
        / output_cfg.get(
            "confusion_matrix",
            "confusion_matrix.csv",
        )
    )

    cm_df.to_csv(
        confusion_matrix_path
    )

    # --------------------------------------------------------
    # TEST RESULTS
    # --------------------------------------------------------

    mean_confidence = float(
        test_confidences.mean()
    )

    test_results = {

        "test_images":
            len(test_targets),

        "test_accuracy":
            float(test_accuracy),

        "test_macro_f1":
            float(test_macro_f1),

        "best_epoch":
            int(best_epoch),

        "best_validation_macro_f1":
            float(best_val_f1),

        "mean_test_confidence":
            mean_confidence,

        "architecture":
            architecture,

        "num_classes":
            num_classes,

        "classes":
            classes,

        "device":
            str(device),
    }

    test_results_path = (
        output_dir
        / output_cfg.get(
            "test_results",
            "test_results.json",
        )
    )

    save_json(
        test_results_path,
        test_results,
    )

    # --------------------------------------------------------
    # TRAINING METADATA
    # --------------------------------------------------------

    metadata = {

        "architecture":
            architecture,

        "image_size":
            image_size,

        "batch_size":
            batch_size,

        "epochs_requested":
            epochs,

        "epochs_completed":
            len(history),

        "learning_rate":
            learning_rate,

        "weight_decay":
            weight_decay,

        "patience":
            patience,

        "seed":
            seed,

        "num_workers":
            num_workers,

        "num_classes":
            num_classes,

        "classes":
            classes,

        "dataset_images":
            len(df),

        "train_images":
            len(train_df),

        "validation_images":
            len(val_df),

        "test_images":
            len(test_df),

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            float(best_val_f1),

        "test_accuracy":
            float(test_accuracy),

        "test_macro_f1":
            float(test_macro_f1),

        "mean_test_confidence":
            mean_confidence,

        "device":
            str(device),

        "manifest":
            str(manifest_path),

        "image_root":
            str(raw_directory),

        "best_model":
            str(best_model_path),
    }

    metadata_path = (
        output_dir
        / output_cfg.get(
            "metadata",
            "training_metadata.json",
        )
    )

    save_json(
        metadata_path,
        metadata,
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"Best epoch:          "
        f"{best_epoch}"
    )

    print(
        f"Best validation F1:  "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Test accuracy:       "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test macro-F1:       "
        f"{test_macro_f1:.4f}"
    )

    print(
        f"Mean test confidence:"
        f" {mean_confidence:.4f}"
    )

    print("\nOutputs:")

    print(
        f"  Best model:        "
        f"{best_model_path}"
    )

    print(
        f"  History:           "
        f"{history_path}"
    )

    print(
        f"  Classification:    "
        f"{classification_report_path}"
    )

    print(
        f"  Confusion matrix:  "
        f"{confusion_matrix_path}"
    )

    print(
        f"  Test results:      "
        f"{test_results_path}"
    )

    print(
        f"  Metadata:          "
        f"{metadata_path}"
    )

    print("\n" + "=" * 72)

    print(
        "BASELINE MODEL READY"
    )

    print(
        "Next stage: model evaluation "
        "and explainability (Grad-CAM)."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()