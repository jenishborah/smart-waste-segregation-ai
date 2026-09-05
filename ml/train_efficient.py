"""
WASTE SEGREGATION AI
EfficientNet-B0 — Cached Dataset Training

Dataset:
    13,196 images
    16 classes

The images are preprocessed once into:
    data/cache_224/train.npz
    data/cache_224/val.npz
    data/cache_224/test.npz

This avoids repeatedly decoding thousands of JPEG/PNG/TIFF
files during every epoch.
"""

import os
import json
import time
import random

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = "/content/smart-waste-segregation-ai"

CACHE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cache_224"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "training_outputs",
    "efficientnet_b0_cached"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 64

EPOCHS = 12

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

PATIENCE = 4

NUM_WORKERS = 2

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# AMP
# ============================================================

USE_AMP = DEVICE.type == "cuda"

if USE_AMP:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("WASTE SEGREGATION AI")
print("EFFICIENTNET-B0 — CACHED DATASET TRAINING")
print("=" * 70)

print()

print("Project:", PROJECT_ROOT)

print("Cache:", CACHE_DIR)

print("Output:", OUTPUT_DIR)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory / 1024**3,
            2
        ),
        "GB"
    )

print()


# ============================================================
# LOAD CLASS INFORMATION
# ============================================================

with open(
    os.path.join(
        CACHE_DIR,
        "class_names.json"
    )
) as f:

    class_names = json.load(f)


with open(
    os.path.join(
        CACHE_DIR,
        "class_to_idx.json"
    )
) as f:

    class_to_idx = json.load(f)


NUM_CLASSES = len(class_names)


print("=" * 70)
print("CLASSES")
print("=" * 70)

for idx, name in enumerate(class_names):

    print(
        f"{idx:2d}  {name}"
    )

print()

print(
    "Number of classes:",
    NUM_CLASSES
)


# ============================================================
# CACHED DATASET
# ============================================================

class CachedDataset(Dataset):

    def __init__(
        self,
        cache_file,
        train=False
    ):

        self.data = np.load(
            cache_file,
            mmap_mode="r"
        )

        self.images = self.data["images"]

        self.labels = self.data["labels"]

        self.train = train

        # ----------------------------------------------------
        # Training augmentation
        # ----------------------------------------------------

        if train:

            self.transform = transforms.Compose([

                transforms.ToPILImage(),

                transforms.RandomResizedCrop(
                    IMAGE_SIZE,
                    scale=(0.80, 1.0)
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.RandomRotation(
                    10
                ),

                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15
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
                    ]
                )
            ])

        # ----------------------------------------------------
        # Validation/test
        # ----------------------------------------------------

        else:

            self.transform = transforms.Compose([

                transforms.ToPILImage(),

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
                    ]
                )
            ])


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, index):

        image = self.images[index]

        label = int(
            self.labels[index]
        )

        image = self.transform(
            image
        )

        return image, label


# ============================================================
# DATASETS
# ============================================================

train_dataset = CachedDataset(
    os.path.join(
        CACHE_DIR,
        "train.npz"
    ),
    train=True
)

val_dataset = CachedDataset(
    os.path.join(
        CACHE_DIR,
        "val.npz"
    ),
    train=False
)

test_dataset = CachedDataset(
    os.path.join(
        CACHE_DIR,
        "test.npz"
    ),
    train=False
)


print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Test:",
    len(test_dataset)
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=(
        NUM_WORKERS > 0
    )
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE * 2,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=(
        NUM_WORKERS > 0
    )
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE * 2,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=(
        NUM_WORKERS > 0
    )
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)


weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)


# Replace classifier

in_features = (
    model.classifier[1].in_features
)

model.classifier[1] = nn.Linear(
    in_features,
    NUM_CLASSES
)


model = model.to(DEVICE)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "Model: EfficientNet-B0"
)

print(
    "Parameters:",
    f"{parameter_count:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=2
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            outputs = model(images)

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

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)


    epoch_loss = (
        running_loss / total
    )

    epoch_acc = (
        correct / total
    )

    return epoch_loss, epoch_acc


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += images.size(0)


    return (
        running_loss / total,
        correct / total
    )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Workers:",
    NUM_WORKERS
)

print(
    "AMP:",
    USE_AMP
)

print()


history = []

best_val_loss = float(
    "inf"
)

best_val_acc = 0.0

best_epoch = 0

patience_counter = 0

training_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_acc = (
        train_one_epoch()
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_loss, val_acc = evaluate(
        val_loader
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )


    current_lr = optimizer.param_groups[
        0
    ]["lr"]


    elapsed = (
        time.time()
        - epoch_start
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history.append({

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy": train_acc,

        "val_loss": val_loss,

        "val_accuracy": val_acc,

        "learning_rate": current_lr,

        "epoch_time_seconds": elapsed

    })


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"  Train Loss: {train_loss:.4f}"
    )

    print(
        f"  Train Acc : {train_acc:.4f}"
    )

    print(
        f"  Val Loss  : {val_loss:.4f}"
    )

    print(
        f"  Val Acc   : {val_acc:.4f}"
    )

    print(
        f"  LR        : {current_lr:.7f}"
    )

    print(
        f"  Time      : {elapsed:.1f}s"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    improved = (
        val_loss < best_val_loss
    )


    if improved:

        best_val_loss = val_loss

        best_val_acc = val_acc

        best_epoch = epoch

        patience_counter = 0


        model_path = os.path.join(
            OUTPUT_DIR,
            "best_model.pth"
        )


        torch.save({

            "model_state_dict":
                model.state_dict(),

            "class_names":
                class_names,

            "class_to_idx":
                class_to_idx,

            "architecture":
                "efficientnet_b0",

            "image_size":
                IMAGE_SIZE,

            "epoch":
                epoch,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_acc

        }, model_path)


        print(
            "  ✓ Best model saved"
        )

    else:

        patience_counter += 1

        print(
            f"  No improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    print()


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)


history_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    ),
    index=False
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(

    os.path.join(
        OUTPUT_DIR,
        "best_model.pth"
    ),

    map_location=DEVICE,

    weights_only=False
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation loss:",
    checkpoint["val_loss"]
)

print(
    "Best validation accuracy:",
    checkpoint["val_accuracy"]
)


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)


all_predictions = []

all_labels = []

test_loss = 0.0

test_total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

        else:

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )


        test_loss += (
            loss.item()
            * images.size(0)
        )

        test_total += images.size(0)


        predictions = outputs.argmax(
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


test_loss /= test_total


all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


test_accuracy = (
    all_predictions
    == all_labels
).mean()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    all_labels,

    all_predictions,

    target_names=class_names,

    output_dict=True,

    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


report_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "classification_report.csv"
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions,

    labels=list(
        range(NUM_CLASSES)
    )
)


cm_df = pd.DataFrame(

    cm,

    index=class_names,

    columns=class_names
)


cm_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)


# ============================================================
# METRICS
# ============================================================

macro_precision = report[
    "macro avg"
]["precision"]

macro_recall = report[
    "macro avg"
]["recall"]

macro_f1 = report[
    "macro avg"
]["f1-score"]


print()

print(
    f"Test Loss:       {test_loss:.4f}"
)

print(
    f"Test Accuracy:   {test_accuracy:.4f}"
)

print(
    f"Macro Precision: {macro_precision:.4f}"
)

print(
    f"Macro Recall:    {macro_recall:.4f}"
)

print(
    f"Macro F1:        {macro_f1:.4f}"
)


# ============================================================
# TEST RESULTS
# ============================================================

test_results = {

    "architecture":
        "efficientnet_b0",

    "test_images":
        int(test_total),

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(test_accuracy),

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "best_epoch":
        int(checkpoint["epoch"]),

    "best_validation_loss":
        float(checkpoint["val_loss"]),

    "best_validation_accuracy":
        float(checkpoint["val_accuracy"])

}


with open(

    os.path.join(
        OUTPUT_DIR,
        "test_results.json"
    ),

    "w"

) as f:

    json.dump(
        test_results,
        f,
        indent=2
    )


# ============================================================
# TRAINING METADATA
# ============================================================

total_time = (
    time.time()
    - training_start
)


metadata = {

    "architecture":
        "efficientnet_b0",

    "dataset":
        "dataset_v1_curated",

    "num_classes":
        NUM_CLASSES,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        EPOCHS,

    "epochs_completed":
        len(history),

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "patience":
        PATIENCE,

    "seed":
        SEED,

    "num_workers":
        NUM_WORKERS,

    "amp":
        USE_AMP,

    "device":
        str(DEVICE),

    "gpu":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),

    "best_epoch":
        int(checkpoint["epoch"]),

    "best_validation_loss":
        float(checkpoint["val_loss"]),

    "best_validation_accuracy":
        float(checkpoint["val_accuracy"]),

    "test_accuracy":
        float(test_accuracy),

    "macro_f1":
        float(macro_f1),

    "total_training_time_seconds":
        float(total_time)

}


with open(

    os.path.join(
        OUTPUT_DIR,
        "training_metadata.json"
    ),

    "w"

) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "Best model:",
    os.path.join(
        OUTPUT_DIR,
        "best_model.pth"
    )
)

print(
    "History:",
    os.path.join(
        OUTPUT_DIR,
        "training_history.csv"
    )
)

print(
    "Classification report:",
    os.path.join(
        OUTPUT_DIR,
        "classification_report.csv"
    )
)

print(
    "Confusion matrix:",
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)

print(
    "Test results:",
    os.path.join(
        OUTPUT_DIR,
        "test_results.json"
    )
)

print()

print(
    f"Final Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Final Macro F1: "
    f"{macro_f1:.4f}"
)

print(
    f"Total training time: "
    f"{total_time / 60:.2f} minutes"
)

print()

print("STATUS: SUCCESS")