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

# 2. Dataset Sources

## 2.1 TrashNet

**Source:**
https://github.com/garythung/trashnet

**Type:** Image classification

**Purpose in this project:** Baseline dataset

### Description

TrashNet is a commonly used waste-image dataset containing images from six waste categories:

* Glass
* Paper
* Cardboard
* Plastic
* Metal
* Trash

The original repository reports **2,527 images**:

| Class     |    Images |
| --------- | --------: |
| Glass     |       501 |
| Paper     |       594 |
| Cardboard |       403 |
| Plastic   |       482 |
| Metal     |       410 |
| Trash     |       137 |
| **Total** | **2,527** |

The images were photographed by placing waste objects on a white posterboard using sunlight and/or room lighting. The repository reports resized images at approximately **512 × 384 pixels**.

### Role

TrashNet will primarily be used to:

1. Establish an initial classification baseline.
2. Test the EfficientNet-B0 training pipeline.
3. Evaluate performance on a relatively controlled dataset.
4. Compare performance against more realistic datasets.

### Limitation

The controlled background and image acquisition conditions may not represent photographs taken by users in real-world environments.

Therefore, TrashNet should **not be treated as sufficient evidence of real-world performance**.

### License

MIT License, according to the source repository.

### Citation

Please cite the original TrashNet repository when using this dataset.

---

# 3. RealWaste

**Source:**
https://archive.ics.uci.edu/dataset/908/realwaste

**Type:** Image classification

**Purpose in this project:** Real-world waste classification and generalization

### Description

RealWaste is a waste-image dataset containing **4,752 images across nine material categories**.

The dataset was collected in an authentic landfill/facility environment, making it useful for studying waste classification under less controlled visual conditions.

The nine categories are:

* Cardboard
* Food Organics
* Glass
* Metal
* Miscellaneous Trash
* Paper
* Plastic
* Textile Trash
* Vegetation

### Role

RealWaste will be used to:

1. Evaluate classification on more realistic waste imagery.
2. Compare against the controlled TrashNet baseline.
3. Investigate generalization across different visual conditions.
4. Potentially contribute to the combined training dataset.

### Importance

RealWaste is particularly important because our intended application involves photographs captured in normal environments rather than objects photographed exclusively against clean backgrounds.

### License

The UCI dataset/source information should be reviewed before redistribution or modification.

### Citation

Use the official UCI RealWaste dataset citation and source information.

---

# 4. PhenomSG Waste Classification Dataset

**Source:**
https://www.kaggle.com/datasets/phenomsg/waste-classification

**Type:** Image classification

**Purpose in this project:** Broader waste coverage, including hazardous and special waste

### Description

The dataset is described as containing **30,000+ labeled images** covering four major waste groups:

* Hazardous
* Non-Recyclable
* Organic
* Recyclable

The dataset documentation describes the following subcategories:

### Hazardous

* Batteries
* Chemical-Waste
* Medical-Waste

### Non-Recyclable

* Plastic-Wrappers
* Styrofoam
* Food-Cups

### Organic

* Food-Waste
* Green-Waste

### Recyclable

* Paper
* Glass
* Plastic-Bottles

### Role

This dataset is particularly relevant to the safety-oriented part of our project.

It can provide additional examples of:

* Batteries
* Chemical waste
* Medical waste
* Organic waste
* Plastic bottles
* Paper
* Glass
* Non-recyclable materials

These categories can help us investigate whether the model can distinguish ordinary waste from materials requiring special handling.

### Important Dataset Note

The dataset's headline describes 30,000+ images, while the currently displayed Kaggle data explorer reports a smaller file count for the downloadable version.

Therefore, the **actual downloaded dataset will be profiled locally** before we use it.

We will not assume the advertised image count is the final number used in our experiments.

### Role in Final Training

The dataset will **not automatically be merged** with the other datasets.

Before inclusion, we will examine:

* Actual image count
* Class distribution
* Image quality
* Duplicate images
* Label consistency
* Visual similarity with other datasets
* Class overlap
* Licensing/provenance
* Suitability for our final taxonomy

### License

MIT License, according to the Kaggle dataset page.

---

# 5. E-Waste Image Dataset

**Source:**
https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset

**Type:** Image classification

**Purpose in this project:** E-waste and special-material coverage

### Description

This dataset contains images of electronic waste divided into ten classes:

1. PCB
2. Player
3. Battery
4. Microwave
5. Mobile
6. Mouse
7. Printer
8. Television
9. Washing Machine
10. Keyboard

The dataset is organized into:

```text
Train/
Test/
Validation/
```

### Role

The dataset will be investigated for strengthening the model's ability to recognize electronic and special waste.

Potential internal mapping:

```text
Mobile
Keyboard
Mouse
Printer
Television
Microwave
Washing Machine
PCB
Player
        |
        v
     E-Waste
```

Battery requires special consideration because it can belong to a different safety/disposal pathway from ordinary electronic devices.

### Important Dataset Note

We will not automatically merge all ten classes into the main model.

First we will determine whether:

* Fine-grained e-waste classes are useful.
* All electronic items should map to a broader `E-Waste` class.
* Battery should be separated as `Battery / Hazardous`.
* The images are visually compatible with the other datasets.

### License

Apache License 2.0, according to the Kaggle dataset page.

---

# 6. TACO — External Robustness Dataset

**Source:**
https://tacodataset.org/

**Type:** Object detection / segmentation / litter dataset

**Purpose in this project:** External generalization and robustness testing

### Description

TACO (Trash Annotations in Context) contains images of litter in diverse real-world environments and provides annotations suitable for computer vision tasks such as detection and segmentation.

### Role

TACO will initially **not be used as a normal training dataset**.

Instead, it will be considered as an external evaluation source.

The purpose is to test whether a model trained on other waste datasets can handle:

* Different backgrounds
* Outdoor environments
* Different lighting
* Occlusion
* Litter in context
* More complex scenes

This helps us evaluate domain shift and real-world robustness.

---

# 7. Own / Real-World Test Images

In addition to public datasets, we plan to create a small independent collection of photographs for final testing.

These images may be captured using:

* Mobile phones
* Different lighting conditions
* Different backgrounds
* Different distances
* Different orientations
* Real household waste environments

### Important Rule

Images used for the final independent test set must **not be used for model training**.

The purpose is to answer:

> Does the trained model work on photographs similar to what an actual user might submit?

---

# 8. Dataset Roles

| Dataset    | Primary Role                 |     Training | Internal Testing | External Testing |
| ---------- | ---------------------------- | -----------: | ---------------: | ---------------: |
| TrashNet   | Baseline                     |          Yes |              Yes |               No |
| RealWaste  | Real-world material data     |          Yes |              Yes |               No |
| PhenomSG   | Broader + hazardous coverage |  Potentially |              Yes |               No |
| E-Waste    | E-waste coverage             |  Potentially |              Yes |               No |
| TACO       | Real-world robustness        | Initially No |               No |              Yes |
| Own Photos | Application validation       | No initially |               No |              Yes |

---

# 9. Planned Experiments

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

# 10. Final ML Taxonomy

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

# 11. ML Classes vs Disposal Categories

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

# 12. Dataset Processing Pipeline

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

# 13. Data Leakage Prevention

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

# 14. Dataset Directory Structure

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

# 15. Dataset Profiling Requirements

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

# 16. Important Principles

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

## 17. Current Status

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
