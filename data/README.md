# Waste Segregation AI — Dataset Documentation

This directory documents the public datasets used for the **AI-Assisted Waste Segregation with Uncertainty-Aware Disposal Guidance** project.

The datasets are used for research, model development, validation, robustness testing, and evaluation. Raw datasets are **not stored in this GitHub repository** due to their size and dataset-specific licensing/provenance requirements.

---

## 1. Dataset Strategy

Our project does not rely on a single dataset.

Different datasets are used for different purposes:

```text
                    PUBLIC DATASETS
                          |
        +-----------------+------------------+
        |                 |                  |
     TrashNet          RealWaste         PhenomSG
        |                 |                  |
        +-----------------+------------------+
                          |
                   Model Development
                          |
                    EfficientNet-B0
                          |
                +---------+---------+
                |                   |
           Internal Test      External Testing
                                    |
                              +-----+------+
                              |            |
                             TACO      Own Photos
```

The **E-Waste Image Dataset** is additionally used to improve coverage of electronic and special waste categories.

The final training composition will be determined only after dataset profiling, class analysis, duplicate checking, quality inspection, and label-mapping analysis.

---


# 2. Dataset Roles

| Dataset    | Primary Role                 |     Training | Internal Testing | External Testing |
| ---------- | ---------------------------- | -----------: | ---------------: | ---------------: |
| TrashNet   | Baseline                     |          Yes |              Yes |               No |
| RealWaste  | Real-world material data     |          Yes |              Yes |               No |
| PhenomSG   | Broader + hazardous coverage |  Potentially |              Yes |               No |
| E-Waste    | E-waste coverage             |  Potentially |              Yes |               No |
| TACO       | Real-world robustness        | Initially No |               No |              Yes |
| Own Photos | Application validation       | No initially |               No |              Yes |

---

# 3. Planned Experiments

The datasets will be evaluated progressively rather than merged immediately.

### Experiment 1 — TrashNet Baseline

```text
TrashNet
    ↓
EfficientNet-B0
    ↓
Classification
```

Purpose:

Establish a reproducible baseline.

---

### Experiment 2 — RealWaste

```text
RealWaste
    ↓
EfficientNet-B0
    ↓
Classification
```

Purpose:

Measure performance on more realistic waste imagery.

---

### Experiment 3 — Combined Conventional Waste

```text
TrashNet + RealWaste
          ↓
   Label Harmonization
          ↓
    EfficientNet-B0
```

Purpose:

Determine whether combining controlled and real-world data improves generalization.

---

### Experiment 4 — Special/Hazardous Coverage

```text
Previous Training Data
          +
       PhenomSG
          ↓
    Label Mapping
          ↓
    EfficientNet-B0
```

Purpose:

Increase coverage of hazardous, organic, recyclable and non-recyclable categories.

---

### Experiment 5 — E-Waste Coverage

```text
Previous Training Data
          +
    E-Waste Dataset
          ↓
    Label Mapping
          ↓
    EfficientNet-B0
```

Purpose:

Improve recognition of electronic and special waste.

---

### Experiment 6 — Uncertainty-Aware Model

The best-performing model will be combined with an uncertainty mechanism using:

* Maximum softmax probability
* Prediction entropy
* Image-quality checks
* Out-of-distribution testing

Low-confidence or unsupported inputs may be routed to:

```text
UNCERTAIN
```

instead of forcing an unreliable classification.

---

### Experiment 7 — External Evaluation

The selected model will be evaluated using:

```text
TACO
Own real-world photographs
OOD / unsupported images
Poor-quality images
```

The purpose is to measure real-world robustness rather than only benchmark performance.

---

# 4. Final ML Taxonomy

The final ML classes are **not yet fixed**.

The preliminary taxonomy under investigation is:

```text
1. Food / Organic
2. Paper
3. Cardboard
4. Plastic
5. Glass
6. Metal
7. Textile
8. E-Waste
9. Battery / Hazardous
10. Medical / Sanitary
11. Chemical / Hazardous
12. Residual / Other
```

This taxonomy will be finalized only after dataset profiling.

---

# 5. ML Classes vs Disposal Categories

The neural network should identify what the item **looks like/materially represents**.

A separate policy layer determines what the user should **do with it**.

Example:

```text
Image
  ↓
ML Model
  ↓
Battery / Hazardous
  ↓
Policy Engine
  ↓
Domestic Hazardous / Special Waste
  ↓
Local disposal guidance
```

This separation is intentional.

A visual classifier should not independently claim that an item is universally recyclable because actual disposal rules can vary by location, collection infrastructure, contamination, and material composition.

---

# 6. Dataset Processing Pipeline

Before training, every dataset will go through:

```text
Raw Dataset
     ↓
Dataset Profiling
     ↓
Image Validation
     ↓
Corrupt Image Removal
     ↓
Duplicate Detection
     ↓
Class Distribution Analysis
     ↓
Label Harmonization
     ↓
Quality Analysis
     ↓
Train / Validation / Test Split
     ↓
Data Augmentation
     ↓
Model Training
```

---

# 7. Data Leakage Prevention

The same image, or near-duplicate versions of the same image, must not appear across training and evaluation sets.

Particular attention will be given to:

* Duplicate images
* Near-duplicate images
* Augmented copies
* Dataset overlap
* Existing train/test splits
* Images originating from the same source

External test datasets will remain isolated from model development whenever possible.

---

# 8. Dataset Directory Structure

The local dataset directory should eventually follow:

```text
data/
│
├── README.md
│
├── raw/
│   ├── trashnet/
│   ├── realwaste/
│   ├── phenomsg/
│   └── e_waste/
│
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
│
└── external/
    ├── taco/
    └── own_photos/
```

Raw public datasets should remain unchanged.

Processed datasets should be generated by reproducible scripts rather than manually moving files.

---

# 9. Dataset Profiling Requirements

Before training the final model, the following information will be recorded for every dataset:

* Total image count
* Number of classes
* Images per class
* Image dimensions
* File formats
* Corrupt images
* Duplicate images
* Near-duplicates
* Class imbalance
* Background characteristics
* Lighting conditions
* Object size
* Occlusion
* Single-item vs multiple-item scenes
* Label quality
* License
* Source/provenance
* Suitability for final taxonomy

---

# 10. Important Principles

### Do not train before profiling

We will inspect the datasets before deciding the final training composition.

### Do not blindly merge datasets

Different datasets may use different definitions for similar labels.

### Do not use external test data for tuning

TACO and the independent real-world test images should provide a more honest estimate of generalization.

### Do not assume "recyclable" is a universal visual class

Recyclability and disposal instructions can depend on local waste-management systems.

### Do not let the model force every image into a known class

The final system should have an **Uncertain** state for ambiguous, unsupported, poor-quality, or potentially out-of-distribution inputs.

---

## 11. Current Status

| Dataset                    | Status                           |
| -------------------------- | -------------------------------- |
| TrashNet                   | Selected                         |
| RealWaste                  | Selected                         |
| PhenomSG                   | Selected for investigation       |
| E-Waste Dataset            | Selected for investigation       |
| TACO                       | Reserved for external evaluation |
| Own Photos                 | Planned                          |
| Final taxonomy             | **Pending profiling**            |
| Final training combination | **Pending profiling**            |

**Next step:** Download the selected datasets locally and perform dataset profiling before creating the final processed dataset.
