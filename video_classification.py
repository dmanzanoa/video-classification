#!/usr/bin/env python
"""
Video Classification for Exercise Analysis
========================================

This module provides a general purpose toolkit for building and training
video classification models using a hybrid convolutional–recurrent
architecture.  It is tailored for problems like exercise recognition
where short clips need to be assigned to one of several movements or
activities.  The default model combines a convolutional backbone for
per‑frame feature extraction with a gated recurrent unit (GRU) to
capture temporal dynamics.  The code is written to be framework agnostic
with respect to the dataset: you supply a CSV describing where your
videos live and what label each belongs to, and optionally enable
automatic class balancing.

Key features
------------
* **CSV‑driven data loading:** specify your dataset using a CSV with
  columns for the file path and label.  Use ``load_dataframe`` to read
  the CSV and optionally oversample classes to mitigate imbalance.
* **Reproducible augmentation:** the optional ``augment_video`` function
  illustrates how to apply random transformations to a sequence of
  frames using the ``vidaug`` library.  You can adapt or remove this
  depending on your own augmentation strategy.
* **Modular model construction:** ``build_model`` creates a
  convolutional feature extractor (EfficientNetB0 by default) and
  stacks a GRU on top.  Swap ``EfficientNetB0`` for another Keras
  application model if desired.
* **Utility functions:** helper routines for loading frames from disk
  (``load_video``), extracting features in batches, and balancing
  training data via random oversampling are included.

This script does not perform end‑to‑end training by itself.  Instead it
defines reusable building blocks that can be composed in your own
training loop or integrated into a larger pipeline.  See the README
for a high level overview and refer to the ``main`` section at the
bottom of this file for a minimal usage example.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2  # type: ignore
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from tensorflow import keras

try:
    from vidaug import augmentors as va  # type: ignore
except ImportError:
    va = None  # augmentation is optional

# -----------------------------------------------------------------------------
# Hyperparameters and defaults
#
# These values can be overridden when calling ``build_model`` or other
# functions.  Adjust them according to your own dataset and experiments.
# -----------------------------------------------------------------------------
IMG_SIZE: int = 224
MAX_SEQ_LENGTH: int = 400
NUM_FEATURES: int = 2560


def load_dataframe(csv_path: str, path_col: str = "path", label_col: str = "label", balance: bool = False) -> pd.DataFrame:
    """Load a dataset description from a CSV file.

    The CSV is expected to contain at least two columns: one with the file
    system path to each video and another with the corresponding class
    label.  Additional columns are ignored.  When ``balance`` is True
    the returned frame is randomly oversampled so that all classes
    contain the same number of samples as the largest class.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file.  Each row should correspond to a single video.
    path_col : str, optional
        Name of the column containing the file path, by default ``"path"``.
    label_col : str, optional
        Name of the column containing the class label, by default ``"label"``.
    balance : bool, optional
        Whether to oversample minority classes to match the size of the
        majority class, by default ``False``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with at least the ``path_col`` and ``label_col`` present.
    """
    df = pd.read_csv(csv_path)
    if balance:
        # Determine the size of the largest class
        max_count = df[label_col].value_counts().max()
        balanced_frames: List[pd.DataFrame] = []
        for label, group in df.groupby(label_col):
            if len(group) < max_count:
                balanced_group = resample(group, replace=True, n_samples=max_count, random_state=42)
            else:
                balanced_group = group
            balanced_frames.append(balanced_group)
        df = pd.concat(balanced_frames).reset_index(drop=True)
    return df[[path_col, label_col]].copy()


def load_video(path: str, max_frames: int = 0, resize: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    """Load a video from disk into a numpy array of frames.

    Frames are centre‑cropped to a square and resized to ``resize``.  If
    ``max_frames`` is non‑zero the returned array will contain at most
    ``max_frames`` frames.

    Parameters
    ----------
    path : str
        Filesystem path to the video.  Any format supported by OpenCV is
        accepted.
    max_frames : int, optional
        Maximum number of frames to return, by default 0 (no limit).
    resize : tuple of int, optional
        Spatial resolution ``(height, width)`` for each frame, by default
        ``(IMG_SIZE, IMG_SIZE)``.

    Returns
    -------
    numpy.ndarray
        A 4‑D tensor of shape ``(num_frames, height, width, 3)`` with dtype
        ``np.uint8``.
    """
    cap = cv2.VideoCapture(path)
    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # centre crop to square
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            frame = frame[top : top + min_dim, left : left + min_dim]
            frame = cv2.resize(frame, resize)
            # convert from BGR to RGB
            frame = frame[:, :, ::-1]
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return np.array(frames, dtype=np.uint8)


def augment_video(frames: np.ndarray) -> np.ndarray:
    """Apply a random augmentation sequence to a sequence of frames.

    This function uses the ``vidaug`` library to compose a random
    assortment of spatial transformations.  If ``vidaug`` is not
    available the input is returned unchanged.

    Parameters
    ----------
    frames : numpy.ndarray
        A 4‑D array of frames with shape ``(num_frames, height, width, 3)``.

    Returns
    -------
    numpy.ndarray
        Augmented frames with the same shape as the input.
    """
    if va is None:
        return frames
    seq = va.Sequential([
        va.SomeOf([
            va.HorizontalFlip(),
            va.VerticalFlip(),
            va.GaussianBlur(random.uniform(1.0, 1.5)),
            va.Pepper(),
            va.Salt(),
            va.Add(random.randint(1, 50)),
            va.Multiply(random.uniform(0.6, 1.4)),
        ], 2)
    ])
    augmented = seq(frames)
    return np.asarray(augmented, dtype=np.uint8)


def build_model(num_classes: int, img_size: int = IMG_SIZE, num_features: int = NUM_FEATURES, max_seq_length: int = MAX_SEQ_LENGTH) -> keras.Model:
    """Construct a CNN–RNN model for video classification.

    A convolutional base (EfficientNetB0) extracts per‑frame features and a
    GRU aggregates them over time.  The final dense layer has size
    ``num_classes`` with softmax activation.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    img_size : int, optional
        Input frame dimension.  Frames are assumed to be ``img_size × img_size``.
    num_features : int, optional
        Dimensionality of the feature extractor output per frame.
    max_seq_length : int, optional
        Maximum number of frames to consider.  Longer sequences will be
        truncated.

    Returns
    -------
    tensorflow.keras.Model
        An uncompiled Keras model ready for training.
    """
    # Backbone for feature extraction
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    preprocess_input = keras.applications.efficientnet.preprocess_input

    # Define inputs: a batch of sequences of frames
    inputs = keras.Input(shape=(None, img_size, img_size, 3), name="video_frames")

    # TimeDistributed layer applies the base model to each frame
    def apply_cnn(frame_batch):
        # collapse time dimension into batch dimension for preprocessing
        b, t, h, w, c = tf.unstack(tf.shape(frame_batch))
        frames_reshaped = tf.reshape(frame_batch, (-1, h, w, c))
        x = preprocess_input(tf.cast(frames_reshaped, tf.float32))
        x = base_model(x)
        # reshape back to (batch, time, features)
        x = tf.reshape(x, (b, t, -1))
        return x

    features = tf.keras.layers.Lambda(apply_cnn, name="frame_features")(inputs)

    # Pad or truncate sequences to max_seq_length
    features = tf.keras.layers.Lambda(lambda x: x[:, :max_seq_length, :], name="truncate_seq")(features)

    # Temporal modelling with a GRU
    x = keras.layers.GRU(num_features, return_sequences=False, name="gru")(features)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(inputs, outputs, name="cnn_rnn_video_classifier")
    return model


def encode_labels(labels: Iterable[str]) -> Tuple[np.ndarray, LabelEncoder]:
    """Encode string labels as integer indices.

    Parameters
    ----------
    labels : Iterable[str]
        An iterable of class labels.

    Returns
    -------
    tuple of numpy.ndarray and sklearn.preprocessing.LabelEncoder
        The encoded labels and the fitted encoder.
    """
    le = LabelEncoder()
    y = le.fit_transform(list(labels))
    return y, le


if __name__ == "__main__":
    # Minimal usage example.  To run a real experiment you should replace
    # ``example.csv`` with your own metadata file and implement a data
    # generator that yields batches of frames and labels.  This demo will
    # simply construct the model for a hypothetical dataset with three
    # classes and print its summary.
    print("Building model for 3 classes...")
    model = build_model(num_classes=3)
    model.summary()
