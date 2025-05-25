# Soil Classification using Deep Learning

This project classifies soil images into different soil types using a Convolutional Neural Network (CNN) based on the ResNet18 architecture. It was developed as part of a Kaggle competition focused on soil classification.

## Project Overview

The goal is to build a model that accurately classifies soil images into classes such as alluvial soil, red soil, black soil, and clay soil. The evaluation metric used is the minimum F1 score across all classes.

## Folder Structure

- `notebooks/`: Contains Jupyter notebooks for training the model (`training.ipynb`) and making predictions (`inference.ipynb`).
- `src/`: Source code for model, data processing, and utility scripts.
- `data/`: Contains scripts for downloading the dataset (`download.sh`). **Note:** The actual data is not stored here.
- `docs/`: Documentation files including architecture diagrams (`architecture.png`) and metrics.
  - `docs/cards/`: Contains project summary files like `project-card.ipynb` and evaluation metrics (`ml-metrics.json`).
