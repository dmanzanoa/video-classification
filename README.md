# Video Classification with CNN–RNN

This repository contains code for building and training a video classifier using a hybrid convolutional–recurrent neural network (CNN–RNN) architecture.  The project originated from a Jupyter/Colab notebook for action recognition on the UCF101 dataset, but has been cleaned up and modularised for reuse in different environments.

## Overview

Videos are sequences of images and therefore contain both spatial and temporal information.  To model these aspects simultaneously, we use a convolutional neural network (CNN) backbone for extracting frame‑level features and a gated recurrent unit (GRU) for modelling dependencies across time.  The default backbone is a pre‑trained EfficientNetB0, but this can be replaced with any image classification model available in TensorFlow’s model zoo.

The code includes utilities for augmenting raw videos using [vidaug](https://github.com/okankop/vidaug).  Random transformations such as flips, blurs, elastic deformations and colour perturbations are applied to increase the robustness of the model.

## Repository structure

| Path | Purpose |
| --- | --- |
| `video_classification.py` | Main Python script containing the model definition, data augmentation helpers and a skeleton training loop. |
| `requirements.txt` | List of Python dependencies needed to run the code. |

## Dataset

The original notebook used a subsampled version of the [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) dataset consisting of short video clips grouped into action categories.  To use this repository, download the videos yourself from the official UCF101 website or any other dataset of interest and organise them into separate folders per class.  The `augment_dataset` function in `video_classification.py` illustrates how to generate additional training examples through augmentation.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your‑username>/video‑classification.git
   cd video‑classification
   ```
2. Create a virtual environment (recommended) and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Ensure you have a working installation of FFmpeg; `scikit‑video` relies on it for reading and writing video files.

## Usage

The `video_classification.py` script is designed to be a starting point.  It defines the model architecture and provides helper functions for video augmentation.  You will need to implement the dataset loading logic yourself.  A typical workflow looks like this:

```python
from video_classification import build_model, augment_dataset

# Step 1: optionally augment your dataset
augment_dataset("data/raw_videos", "data/augmented", n_per_video=10)

# Step 2: prepare a tf.data.Dataset that yields batches of
# shape (batch_size, time_steps, height, width, 3) and labels
train_dataset = ...  # TODO: implement
val_dataset = ...    # TODO: implement

# Step 3: build and compile the model
model = build_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Step 4: train
model.fit(train_dataset, validation_data=val_dataset, epochs=50)
```

## Example application: fall‑risk assessment for older adults

An interesting application of video classification to healthcare is described in the book chapter *“Exploring a Deep Learning Approach for Video Analysis Applied to Older Adults Fall Risk”*.  In that study the authors recorded older adults performing various exercises and developed eight deep‑learning models—including 3D convolutional networks and recurrent architectures—to classify the recorded movements and identify which exercises correlate with higher risk of falling.  According to their report, the models achieved accuracy values between **71 % and 89 %** depending on the specific exercise.  This demonstrates how video analysis can support assessments of functional mobility and highlights potential future directions for projects like this.

## Credits

The original implementation and description were created by Sayak Paul in a Colab notebook for the TensorFlow tutorials.  This repository adapts that work into a standalone Python project suitable for further experimentation and extension.
