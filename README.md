# Sign Language Finger Alphabet Classification using Transfer Learning

## Overview

This project implements a computer vision classification pipeline using transfer learning with EfficientNetB0 to recognize finger alphabet images. The model is trained and evaluated using TensorFlow/Keras and demonstrates strong generalization performance with minimal class confusion.

The workflow covers data loading, preprocessing, dataset serialization, model training, and evaluation, all implemented in a reproducible Jupyter notebook.

## Dataset

Classes: Sign Language Finger datasets -> A, B, C

Structure: Images are organized into class-specific directories

## Pipeline:

Images are loaded using image_dataset_from_directory

Training and validation splits are created automatically

Datasets are saved as tf.data.Dataset objects for faster reuse

Raw image data is excluded from the repository to keep it lightweight and reproducible.

## Model Architecture

Backbone: EfficientNetB0 (ImageNet pretrained)

Strategy: Transfer learning (feature extractor frozen)

Classifier Head: Custom dense layers on top of the backbone

Loss Function: Sparse categorical cross-entropy

Optimizer: Adam / AdamW

Framework: TensorFlow & Keras

##### This approach leverages pretrained visual features while keeping training efficient and stable.

## Results & Evaluation

##### Model performance was evaluated using a confusion matrix on the validation set.

### Key results:

Validation Accuracy: 97.8%

Class B: Perfect classification (100% precision and recall)

Classes A & C: Only minor mutual confusion (3 total misclassifications)

Confusion Matrix Analysis:
The confusion matrix indicates strong class separability and robust feature extraction. All errors are systematic rather than random, suggesting visual similarity between classes rather than model instability.


## Key Technical Highlights:

Robust path handling using pathlib for reproducibility

Dataset caching and prefetching for performance optimization

Offline-safe pretrained weight loading

Clean separation of raw data, processed datasets, and experiments

Notebook designed to run end-to-end without manual intervention


## Future Improvements:

Fine-tuning upper EfficientNet layers

Data augmentation for visually similar classes

Support for additional alphabet classes

Conversion to a Streamlit or FastAPI inference app


## Author:
Azumi Muhammed
Junior Data Scientist
