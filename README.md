# TezHack 2026 — Smart Waste AI by Encryptos

## Waste Material Classification for Better Segregation

AI-assisted waste segregation system that identifies waste materials from images, estimates prediction reliability, and provides practical disposal guidance.

## Core Approach

The system combines:

- Public waste-image datasets
- Lightweight transfer learning
- EfficientNet-B0
- Image preprocessing and augmentation
- Confidence and entropy-based uncertainty detection
- "Uncertain" handling for unclear/OOD images
- Safety-aware handling of e-waste and hazardous/special waste
- Rule-based disposal guidance
- Confusion matrix and prediction analysis
- Real-world/generalization testing

## Research Focus

We will investigate:

1. Whether a lightweight pretrained vision model can classify common waste materials effectively.
2. Whether combining controlled and real-world datasets improves generalization.
3. Whether uncertainty detection reduces overconfident incorrect predictions.
4. Whether separating ML classification from disposal policy makes the system safer and easier to adapt to local rules.

## Primary Model

EfficientNet-B0 pretrained on ImageNet.

## Dataset Strategy

- TrashNet — controlled baseline
- RealWaste — real-world training/validation
- TACO — external real-world robustness testing

## User-Facing Categories

1. Organic / Wet Waste
2. Dry Recyclable Waste
3. General / Residual Waste
4. E-Waste
5. Domestic Hazardous / Special Waste
6. Sanitary Waste
7. Uncertain

## Safety Principle

The system does not certify hazardous material, guarantee recyclability, or replace local municipal waste-handling instructions.

For hazardous, electronic, chemical, medical, sharp, or other special waste, users should follow authorized local guidance.

## MVP Stack

- Python
- PyTorch
- EfficientNet-B0
- Streamlit
- GitHub
- JSON-based disposal rules

## Team Details

Team Name: Encryptos
Team Lead: Barasha Das 
Members: Jenish A. Borah, Mayur Mudoi

