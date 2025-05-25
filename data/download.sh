#!/bin/bash

# Dataset or competition or notebook slug
# Examples:
# dataset: 'annam-ai/soilclassification'
# competition: 'annam-ai/soilclassification'
# notebook: 'tharunbabu20/soil-classification-project'

# === Example: Download a competition dataset ===
KAGGLE_COMPETITION="soil-classification"
TARGET_DIR="./data"

echo "Downloading dataset from competition: $KAGGLE_COMPETITION"
mkdir -p "$TARGET_DIR"
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$TARGET_DIR" --force

echo "Unzipping files..."
unzip -o "$TARGET_DIR/$KAGGLE_COMPETITION.zip" -d "$TARGET_DIR"
rm "$TARGET_DIR/$KAGGLE_COMPETITION.zip"

echo "Download complete. Files saved to $TARGET_DIR"
