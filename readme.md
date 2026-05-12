# CNN Mask / No-Mask Classifier

This repository contains a convolutional neural network (CNN) model for classifying whether a person is wearing a mask or not.

## Overview

- Input: face images or frames from a camera feed
- Output: binary classification: `mask` or `no-mask`
- Model: CNN-based image classifier trained on a labeled dataset of masked and unmasked faces

## Directory Structure

- `data/` - dataset files and image folders
- `models/` - saved model weights and checkpoints
- `notebooks/` - training and evaluation notebooks (if included)
- `src/` - model, preprocessing, and utility scripts

## Setup

1. Create a Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Training

1. Prepare the dataset with two classes: `mask` and `no_mask`.
2. Preprocess images to a consistent size, normalize pixel values, and split into training/testing sets.
3. Train the CNN model using the training data.
4. Save the trained model weights for later inference.

## Inference

1. Load the saved model.
2. Preprocess a new image or camera frame.
3. Run the model and interpret the output label.

## Evaluation

- Track accuracy, precision, recall, and loss during training.
- Validate the model on a held-out test set.
- Use confusion matrix visualization to inspect classification performance.

## Notes

- A balanced dataset helps improve classifier reliability.
- Data augmentation can reduce overfitting.
- Use transfer learning from a pretrained CNN if dataset size is limited.
