# Smart Waste Segregation AI

## ENCRYPTOS — TezHack 2026

**Problem Statement:** ML06 — Waste Material Classification for Better Segregation  
**Challenge Card:** ML-C12 — Why This Result?

Smart Waste Segregation AI is an explainable computer-vision system that classifies waste from an uploaded image and goes beyond a simple class label by showing **why the model made the prediction**.

The system combines:

**Image Quality Validation → EfficientNet-B0 → Prediction + Confidence → Grad-CAM → Human-readable Explanation → Recycling Guidance**

---

# Features

- Waste image upload through a local web interface
- Basic image-quality validation
- EfficientNet-B0 waste classification
- 16-class waste taxonomy
- Prediction confidence
- Top-3 predictions
- Real Grad-CAM visualization
- "Why This Result?" explanation
- Simple human-readable reasoning
- Recycling/segregation guidance
- Clean ENCRYPTOS web interface
- Local Flask backend
- Modular ML pipeline suitable for future API deployment

---

# Project Structure

```text
smart-waste-segregation-ai/
│
├── app.py
├── ml/
│   ├── pipeline.py
│   ├── interface.py
│   ├── explain.py
│   └── export_onnx.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── data/
│   └── raw/
├── uploads/
├── web_uploads/
├── training_outputs/
│   ├── efficientnet_b0_cached/
│   │   ├── best_model.pth
│   │   └── class_names.json
│   ├── explanations/
│   └── pipeline_results/
├── requirements.txt
└── README.md
```

---

# 1. Dataset Sources

The project uses multiple waste-image datasets rather than relying on a single source.

The main public sources are:

1. TrashNet
2. RealWaste
3. PhenoMSg Waste Classification Dataset
4. E-Waste Image Dataset
5. Independent real-world test images

---

## 1.1 TrashNet

**Source:** https://github.com/garythung/trashnet

**Type:** Image classification

**Purpose:** Baseline dataset

TrashNet contains six waste categories:

- Glass
- Paper
- Cardboard
- Plastic
- Metal
- Trash

The original repository reports **2,527 images**.

| Class | Images |
|---|---:|
| Glass | 501 |
| Paper | 594 |
| Cardboard | 403 |
| Plastic | 482 |
| Metal | 410 |
| Trash | 137 |
| **Total** | **2,527** |

The images were photographed by placing waste objects on a white posterboard using sunlight and/or room lighting. The repository reports resized images at approximately **512 × 384 pixels**.

### Role

TrashNet is used to:

1. Establish an initial classification baseline.
2. Test the EfficientNet-B0 training pipeline.
3. Evaluate performance on a relatively controlled dataset.
4. Compare performance against more realistic datasets.

### Limitation

The controlled background and image acquisition conditions may not represent photographs taken by users in real-world environments.

Therefore, TrashNet should **not be treated as sufficient evidence of real-world performance**.

### License

MIT License, according to the source repository.

---

## 1.2 RealWaste

**Source:** https://archive.ics.uci.edu/dataset/908/realwaste

**Type:** Image classification

**Purpose:** Real-world waste classification and generalization

RealWaste contains **4,752 images across nine material categories**:

- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

The dataset was collected in an authentic landfill/facility environment, making it useful for studying waste classification under less controlled visual conditions.

### Role

RealWaste is used to:

1. Evaluate classification on more realistic waste imagery.
2. Compare against the controlled TrashNet baseline.
3. Investigate generalization across different visual conditions.
4. Potentially contribute to the combined training dataset.

### Importance

The intended application involves photographs captured in normal environments rather than objects photographed exclusively against clean backgrounds.

### License

Review the UCI dataset/source information before redistribution or modification.

---

## 1.3 PhenoMSg Waste Classification Dataset

**Source:** https://www.kaggle.com/datasets/phenomsg/waste-classification

**Type:** Image classification

**Purpose:** Broader waste coverage, including hazardous and special waste

The dataset is described as containing **30,000+ labeled images** covering four major waste groups:

- Hazardous
- Non-Recyclable
- Organic
- Recyclable

Documented subcategories include:

### Hazardous

- Batteries
- Chemical-Waste
- Medical-Waste

### Non-Recyclable

- Plastic-Wrappers
- Styrofoam
- Food-Cups

### Organic

- Food-Waste
- Green-Waste

### Recyclable

- Paper
- Glass
- Plastic-Bottles

### Important Dataset Note

The headline describes 30,000+ images, while the currently displayed Kaggle data explorer reports a smaller file count for the downloadable version.

Therefore, the actual downloaded dataset should be profiled locally before being used in experiments.

We do **not** assume that the advertised image count is the final number used in experiments.

### Role in Final Training

The dataset is not automatically merged with the other datasets.

Before inclusion, it should be examined for:

- Actual image count
- Class distribution
- Image quality
- Duplicate images
- Label consistency
- Visual similarity with other datasets
- Class overlap
- Licensing/provenance
- Suitability for the final taxonomy

### License

MIT License, according to the Kaggle dataset page.

---

## 1.4 E-Waste Image Dataset

**Source:** https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset

**Type:** Image classification

**Purpose:** E-waste and special-material coverage

The dataset contains electronic waste divided into ten classes:

- PCB
- Player
- Battery
- Microwave
- Mobile
- Mouse
- Printer
- Television
- Washing Machine
- Keyboard

The dataset is organized into:

```text
Train/
Test/
Validation/
```

### Potential internal grouping

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

The ten E-Waste classes are **not automatically treated as ten independent classes in the final model**.

Before inclusion, the project considers:

- Whether fine-grained e-waste classes are useful.
- Whether electronic items should map to a broader E-Waste class.
- Whether Battery should be separated as Battery / Hazardous.
- Whether the images are visually compatible with the other datasets.
- Whether the resulting taxonomy remains consistent.

### License

Apache License 2.0, according to the Kaggle dataset page.

---

## 1.5 Own / Real-World Test Images

In addition to public datasets, the project plans to create a small independent collection of photographs for final testing.

These images may be captured using:

- Mobile phones
- Different lighting conditions
- Different backgrounds
- Different distances
- Different orientations
- Real household waste environments

### Purpose

The independent test set is intended to answer:

> Does the trained model work on photographs similar to what an actual user might submit?

### Important Rule

Images used for the final independent test set must **not** be used for model training.

---

# 2. Dataset Preparation

The datasets are not blindly merged together.

Before training, datasets are examined for:

- Class overlap
- Label consistency
- Duplicate images
- Image quality
- Class imbalance
- Different naming conventions
- Different visual acquisition conditions
- Suitability for the final taxonomy

Where appropriate, related classes from different datasets can be mapped into a common classification taxonomy.

---

# 3. Final Classification Space

The current Smart Waste Segregation AI model operates on a **16-class waste taxonomy**:

1. Battery
2. Cardboard
3. Electronic Component
4. Electronic Device
5. Food Organics
6. Glass
7. Hazardous Waste
8. Large Electronic Appliance
9. Metal
10. Organic Stream
11. Paper
12. Plastic
13. Recyclable Stream
14. Residual
15. Textile
16. Vegetation

Different source datasets contain different taxonomies, so source-dataset labels are not necessarily identical to the final model labels.

---

# 4. Dataset Download

The project dataset is available separately through Google Drive.

**Google Drive Dataset:**

```text
PASTE YOUR GOOGLE DRIVE DATASET LINK HERE
```

Download the dataset and place the extracted data under:

```text
data/raw/
```

Example:

```text
data/
└── raw/
    ├── trashnet/
    ├── realwaste/
    ├── phenomsg/
    ├── e-waste/
    └── ...
```

The exact directory names may vary depending on how the datasets are downloaded and prepared.

---

# 5. Dataset Attribution

Primary dataset sources:

### TrashNet
https://github.com/garythung/trashnet

### RealWaste
https://archive.ics.uci.edu/dataset/908/realwaste

### PhenoMSg Waste Classification Dataset
https://www.kaggle.com/datasets/phenomsg/waste-classification

### E-Waste Image Dataset
https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset

Please refer to the original dataset pages for authorship, licensing, citation, and redistribution requirements.

---

# 6. Model

## EfficientNet-B0

The project uses **EfficientNet-B0** as the primary waste-image classification model.

### Input

```text
224 × 224 RGB image
```

### Preprocessing

1. RGB conversion
2. Resize to 224 × 224
3. Tensor conversion
4. ImageNet normalization

Mean:

```text
[0.485, 0.456, 0.406]
```

Standard deviation:

```text
[0.229, 0.224, 0.225]
```

EfficientNet-B0 provides a practical balance between classification capability, model size, and computational requirements.

---

# 7. Model Evaluation

| Metric | Result |
|---|---:|
| Test Accuracy | **89.49%** |
| Macro F1 | **0.892** |

These are reported evaluation metrics for the trained model.

---

# 8. Example Inference

Representative test image:

```text
Plastic_821.jpg
```

Result:

```text
Prediction: Plastic
Confidence: 96.90%
```

Top predictions approximately:

```text
1. Plastic             96.90%
2. Metal                0.47%
3. Recyclable Stream    0.36%
```

**Important:** 96.90% is the confidence of this individual prediction, not the overall model accuracy.

---

# 9. Image Quality Validation

Before classification, the web application performs a basic image-quality validation stage.

It checks:

- Whether an image was provided
- Whether the image format is valid
- Whether the image can be opened
- Whether the image is extremely small
- Whether the image is obviously unusable

For unsuitable images, the application can show:

> This image isn't clear enough to classify. Try uploading a brighter, closer photo of the waste item.

This stage is intentionally lightweight and can be improved later.

---

# 10. Explainability — ML-C12 "Why This Result?"

A central part of the project is the **ML-C12 — Why This Result?** concept.

A conventional classifier may provide:

```text
Plastic
96.90%
```

Our system also provides an explanation of the prediction.

---

## 10.1 Grad-CAM

The project implements **real Grad-CAM** using the trained model's activations and gradients.

The generated visualization highlights image regions that had stronger influence on the selected prediction.

The web application displays the actual generated Grad-CAM visualization.

It is not a manually created or randomly generated heatmap.

### Important limitation

Grad-CAM is an activation-based visual explanation. It should not be interpreted as a complete causal explanation of the model's decision.

---

## 10.2 Human-readable explanation

The system also provides a simple explanation alongside the technical visualization.

### Why This Result?

**In simple words:**

> The AI sees strong visual patterns that match what it learned from plastic waste during training. It is quite confident that this item belongs to the Plastic category.

### What is the AI looking at?

> The highlighted areas in the Grad-CAM image show which parts of the photo had the strongest influence on the prediction. Warmer areas mean the model paid more attention there.

---

# 11. Recycling Guidance

After classification, the system provides simple guidance related to the predicted category.

Example for plastic:

> Empty and rinse the plastic item when appropriate, then place it in the designated plastic recycling stream. Follow your local recycling rules.

Local recycling rules can differ, so the application does not claim that one disposal rule applies everywhere.

---

# 12. Web Application

The current working prototype is a local Flask web application.

Start it with:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Current flow

```text
Upload Image
      ↓
Image Quality Validation
      ↓
Analyze
      ↓
EfficientNet-B0
      ↓
Prediction + Confidence
      ↓
Top Predictions
      ↓
Why This Result?
      ↓
Real Grad-CAM
      ↓
Recycling Guidance
```

---

# 13. Web Demo Features

- ENCRYPTOS branding
- TezHack 2026 branding
- ML06 problem statement
- ML-C12 "Why This Result?" concept
- Image upload
- Image preview
- Image quality validation
- Analyze/Classify action
- Loading state
- Prediction result
- Confidence percentage
- Top-3 predictions
- Human-readable explanation
- Grad-CAM visualization
- Recycling guidance
- Try Another Image

---

# 14. Installation

## Requirements

Recommended environment:

- Python 3.10+
- PyTorch
- Torchvision
- Flask
- Pillow
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 15. Running the Demo

From the project root:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Upload a waste image and select **Analyze**.

The application runs the existing inference and explanation pipeline.

---

# 16. Command-Line Inference

The inference interface can also be used directly:

```bash
python ml/interface.py "data/raw/realwaste/Plastic/Plastic_821.jpg"
```

The interface loads the trained EfficientNet-B0 checkpoint and returns the top predictions.

---

# 17. Trained Model

The trained model checkpoint is stored under:

```text
training_outputs/efficientnet_b0_cached/best_model.pth
```

The model is an **EfficientNet-B0** classifier.

The web application uses the trained model for inference and does not retrain it during normal operation.

---

# 18. Explanation Outputs

Generated explanation images and pipeline outputs are stored under:

```text
training_outputs/explanations/
```

and:

```text
training_outputs/pipeline_results/
```

Exact filenames depend on the processed image and prediction.

---

# 19. Architecture

```text
                  USER
                   |
                   v
          Web Interface
        HTML / CSS / JS
                   |
                   v
             Flask app.py
                   |
                   v
           ML Pipeline
          ml/pipeline.py
                   |
          +--------+--------+
          |                 |
          v                 v
   EfficientNet-B0      Validation
          |
          v
    Prediction
          |
          +-------------------+
          |                   |
          v                   v
      Confidence          Grad-CAM
          |                   |
          +---------+---------+
                    |
                    v
             Why This Result?
                    |
                    v
          Recycling Guidance
```

The architecture can later evolve into:

```text
Frontend
   |
   v
REST API
   |
   v
ML Service
   |
   +-- Classifier
   +-- Explainability
   +-- Guidance
```

---

# 20. Important Design Decisions

### No model retraining during inference

The web application uses the existing trained model.

### No fake predictions

Predictions and confidence values shown in the application come from actual model inference.

### No fake Grad-CAM

The explanation visualization is generated using the implemented Grad-CAM mechanism.

### Explainability is part of the product

The project is designed around the ML-C12 question:

> **Why This Result?**

rather than only displaying a classification label.

---

# 21. Limitations

The current prototype has several limitations:

- Performance can vary with lighting, background, orientation, and image quality.
- Multiple waste objects in one image can make classification harder.
- Grad-CAM is an activation visualization rather than a complete causal explanation.
- Recycling guidance is general and local rules should be followed.
- Current image-quality validation is intentionally basic.
- The current application is a local prototype rather than a production deployment.
- Dataset differences can introduce domain shift between training and real-world images.

---

# 22. Future Work

Potential improvements include:

### Better image-quality validation

Develop stronger checks for blur, lighting, framing, and object visibility.

### Class-wise evaluation

Add detailed per-class precision, recall, F1 scores, and confusion matrices.

### Multi-object detection

Extend the system to identify multiple waste objects within a single photograph.

### Mobile optimization

Optimize the model and inference pipeline for mobile devices.

### API deployment

Expose the ML pipeline through a REST API for integration with other applications.

### Localized recycling guidance

Adapt disposal recommendations to local municipal and regional recycling rules.

### Larger real-world test set

Collect more independent user-like images across different:

- phones
- backgrounds
- lighting conditions
- distances
- orientations

---

# 23. Hackathon Context

## Team

**ENCRYPTOS**

## Event

**TezHack 2026**

## Problem Statement

**ML06 — Waste Material Classification for Better Segregation**

## Challenge Card

**ML-C12 — Why This Result?**

### Core Idea

Traditional image classification can answer:

> **What is this?**

ENCRYPTOS aims to answer:

> **What is this? Why did the AI think so? And what should I do with it?**

---

# 24. Project Highlights

| Highlight | Value |
|---|---:|
| Public Dataset Sources | **4** |
| Waste Classes | **16** |
| Test Accuracy | **89.49%** |
| Macro F1 | **0.892** |
| Representative Plastic Confidence | **96.90%** |

The 96.90% figure is an individual prediction confidence and should not be interpreted as the overall model accuracy.

---

# 25. Dataset Download

The project dataset is available separately through Google Drive.

**Google Drive Dataset:**

```text
https://drive.google.com/drive/folders/1oteT82WYiR77Ukyog1wQvpyh_OLVtvgq?usp=sharing
```

Download the dataset and place the extracted data under:

```text
data/raw/
```

---

# 26. Dataset Attribution and Licensing

This project uses publicly available datasets from multiple sources.

Please refer to each original dataset page for current licensing, authorship, citation, and redistribution requirements.

- TrashNet: https://github.com/garythung/trashnet
- RealWaste: https://archive.ics.uci.edu/dataset/908/realwaste
- PhenoMSg: https://www.kaggle.com/datasets/phenomsg/waste-classification
- E-Waste Image Dataset: https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset

Users should verify the current licensing and redistribution terms of each source before redistributing datasets.

---

# 27. Reproducibility Notes

For reproducible experiments:

1. Keep training and independent test images separate.
2. Do not place independent test images into training directories.
3. Preserve the class mapping used by the trained checkpoint.
4. Use the same preprocessing configuration expected by EfficientNet-B0.
5. Record dataset versions and preprocessing changes.
6. Do not compare individual prediction confidence directly with overall test accuracy.

---

# 28. Acknowledgements

We acknowledge the authors and maintainers of the public datasets used during the development and evaluation of this project.

This project was developed as part of **TezHack 2026** under:

**ML06 — Waste Material Classification for Better Segregation**

with the explainability focus:

**ML-C12 — Why This Result?**

---

# 29. License and Dataset Disclaimer

The source code of this project and the datasets used by it may have different licensing terms.

Dataset licenses remain with their respective authors and publishers.

Before redistributing any dataset, verify the original license and usage conditions.

The Google Drive dataset link provided in this repository should therefore be used in accordance with the respective dataset licenses.

---

# ENCRYPTOS

## Smart Waste Segregation AI

> **Don't just tell me what it is. Tell me why.**

**TezHack 2026 · ML06 · ML-C12**
