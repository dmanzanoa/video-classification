# Exercise Video Classification

This repository contains code for building and training a video classifier for exercise recognition using a CNN–RNN architecture. It was inspired by the deep learning study "Exploring a Deep Learning Approach for Video Analysis Applied to Older Adults Fall Risk," where short videos of older adults performing exercises were used to assess fall risk. The provided code has been refactored into reusable functions so it can be applied to any collection of labelled videos organised via a CSV metadata file.

## Overview

Videos contain both spatial and temporal information. To model these aspects simultaneously, the code uses a convolutional neural network (CNN) backbone to extract frame‑level features and a gated recurrent unit (GRU) to model dependencies across time. The default backbone is EfficientNetB0 pretrained on ImageNet, but you can substitute any suitable Keras application.

Augmentation utilities are provided via [`vidaug`](https://github.com/okankop/vidaug) to apply random flips, blurs, noise and brightness shifts, helping improve robustness.

## Repository structure

| Path | Purpose |
| --- | --- |
| `video_classification.py` | Main Python module containing functions for loading datasets from CSVs, optional class balancing, video loading and augmentation, and a model builder that constructs a CNN–RNN classifier. |
| `requirements.txt` | List of Python dependencies needed to run the code. |
| `README.md` | Project documentation and usage instructions. |

## Dataset

The code expects a CSV file with at least two columns: a `path` column pointing to each video file and a `label` column containing the class name. Use the `load_dataframe` function to read this CSV and optionally oversample classes to mitigate imbalance. Videos should be organised anywhere on disk; there is no prescribed folder hierarchy.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/video-classification.git
   cd video-classification
   ```
2. Create a virtual environment (recommended) and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

Below is a minimal example of how to use the library to build a model. You will need to implement frame extraction and batching for your specific dataset.

```python
from video_classification import load_dataframe, load_video, augment_video, build_model, encode_labels
import numpy as np
import tensorflow as tf

# Load metadata from CSV
df = load_dataframe("train_metadata.csv", balance=True)
paths = df["path"].tolist()
labels = df["label"].tolist()

# Encode string labels to integers
y, encoder = encode_labels(labels)

# Build model for the number of classes in your dataset
num_classes = len(encoder.classes_)
model = build_model(num_classes)

# TODO: implement a generator that yields batches of (frames, labels)
# frames = [load_video(p) for p in paths] or create an on-the-fly loader
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model.fit(data_generator, epochs=20, validation_data=val_generator)
```

## Background

In the work "Exploring a Deep Learning Approach for Video Analysis Applied to Older Adults Fall Risk" the authors recorded older adults performing a battery of exercises and developed eight different deep‑learning models—including CNN and GRU architectures—to classify the movements.  Their models achieved between **71 % and 89 % accuracy** depending on the exercise【87114770475911†screenshot】.  This repository adapts that approach into a general codebase that can be used as a starting point for similar video‑based exercise classification projects.

## Credits

Original ideas were inspired by the TensorFlow action recognition tutorial by Sayak Paul and extended to the fall‑risk assessment context.  The present version refactors that notebook into a standalone Python module.
