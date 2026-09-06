# 1. Dataset Sources

## 1.1 TrashNet

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
https://github.com/garythung/trashnet


---

2. RealWaste

Source:
https://archive.ics.uci.edu/dataset/908/realwaste

Type: Image classification

Purpose in this project: Real-world waste classification and generalization

Description

RealWaste is a waste-image dataset containing 4,752 images across nine material categories.

The dataset was collected in an authentic landfill/facility environment, making it useful for studying waste classification under less controlled visual conditions.

The nine categories are:

Cardboard
Food Organics
Glass
Metal
Miscellaneous Trash
Paper
Plastic
Textile Trash
Vegetation
Role

RealWaste will be used to:

Evaluate classification on more realistic waste imagery.
Compare against the controlled TrashNet baseline.
Investigate generalization across different visual conditions.
Potentially contribute to the combined training dataset.
Importance

RealWaste is particularly important because our intended application involves photographs captured in normal environments rather than objects photographed exclusively against clean backgrounds.

License

The UCI dataset/source information should be reviewed before redistribution or modification.

Citation

Use the official UCI RealWaste dataset citation and source information.

3. PhenomSG Waste Classification Dataset

Source:
https://www.kaggle.com/datasets/phenomsg/waste-classification

Type: Image classification

Purpose in this project: Broader waste coverage, including hazardous and special waste

Description

The dataset is described as containing 30,000+ labeled images covering four major waste groups:

Hazardous
Non-Recyclable
Organic
Recyclable

The dataset documentation describes the following subcategories:

Hazardous
Batteries
Chemical-Waste
Medical-Waste
Non-Recyclable
Plastic-Wrappers
Styrofoam
Food-Cups
Organic
Food-Waste
Green-Waste
Recyclable
Paper
Glass
Plastic-Bottles
Role

This dataset is particularly relevant to the safety-oriented part of our project.

It can provide additional examples of:

Batteries
Chemical waste
Medical waste
Organic waste
Plastic bottles
Paper
Glass
Non-recyclable materials

These categories can help us investigate whether the model can distinguish ordinary waste from materials requiring special handling.

Important Dataset Note

The dataset's headline describes 30,000+ images, while the currently displayed Kaggle data explorer reports a smaller file count for the downloadable version.

Therefore, the actual downloaded dataset will be profiled locally before we use it.

We will not assume the advertised image count is the final number used in our experiments.

Role in Final Training

The dataset will not automatically be merged with the other datasets.

Before inclusion, we will examine:

Actual image count
Class distribution
Image quality
Duplicate images
Label consistency
Visual similarity with other datasets
Class overlap
Licensing/provenance
Suitability for our final taxonomy
License

MIT License, according to the Kaggle dataset page.

4. E-Waste Image Dataset

Source:
https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset

Type: Image classification

Purpose in this project: E-waste and special-material coverage

Description

This dataset contains images of electronic waste divided into ten classes:

PCB
Player
Battery
Microwave
Mobile
Mouse
Printer
Television
Washing Machine
Keyboard

The dataset is organized into:

Train/
Test/
Validation/
Role

The dataset will be investigated for strengthening the model's ability to recognize electronic and special waste.

Potential internal mapping:

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

Battery requires special consideration because it can belong to a different safety/disposal pathway from ordinary electronic devices.

Important Dataset Note

We will not automatically merge all ten classes into the main model.

First we will determine whether:

Fine-grained e-waste classes are useful.
All electronic items should map to a broader E-Waste class.
Battery should be separated as Battery / Hazardous.
The images are visually compatible with the other datasets.
License

Apache License 2.0, according to the Kaggle dataset page.

5. Own / Real-World Test Images 

In addition to public datasets, we plan to create a small independent collection of photographs for final testing.

These images may be captured using:

Mobile phones
Different lighting conditions
Different backgrounds
Different distances
Different orientations
Real household waste environments
Important Rule

Images used for the final independent test set must not be used for model training.

The purpose is to answer:

Does the trained model work on photographs similar to what an actual user might submit?
