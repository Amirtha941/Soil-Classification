# 🌱 Soil Classification using Deep Learning

> **Kaggle Competition**: Soil Classification  
> **Team Name**: SoilClassifiers  
> **Affiliation**: Annam.ai, IIT Ropar  
> **Leaderboard Rank**: 40  
> **Minimum F1 Score**: 1.0  

---

## 📌 Objective

This project aims to classify images of soil into one of seven categories using a deep learning model. The solution was developed as part of the Soil Classification competition on Kaggle, where evaluation is based on the **minimum F1 score across all classes**.

---

## 🧠 Approach

- **Pretrained CNN Model**: We use ResNet18 pretrained on ImageNet for transfer learning.
- **Data Augmentation**: Random flip, rotation, and color jitter to improve generalization.
- **Metric Focus**: Optimized for **minimum F1 score** to ensure every class performs well.
- **Modular Design**: All logic is broken into reusable components.

---

## 🗂️ Project Structure

soil-classification/
│
├── /src/ # Core logic
│ ├── PreProcessing.py # Image transforms
│ └── PostProcessing.py # Evaluation logic
│
├── /notebooks/ # Development notebooks
│ ├── training.ipynb
│ └── inference.ipynb
│
├── /docs/ # Documentation & assets
│ ├── architecture.png
│ └── /cards/
│ ├── project-card.ipynb # Summary of project
│ └── ml-metrics.json
│
├── /data/
│ └── download.sh # Script to download dataset from Kaggle
│
├── README.md # Project summary
└── requirements.txt # Dependencies



## 🛠️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/<your-username>/soil-classification.git

cd soil-classification

2. Install dependencies

pip install -r requirements.txt

3. Download the dataset
Make sure you have the Kaggle API set up. Then run:

data/download.sh
This will automatically download and unzip the dataset into the data/ folder.

🚀 How to Run
Train the model:
Open and run the cells in:


notebooks/training.ipynb
Run inference on test data:

notebooks/inference.ipynb
📊 Results

✅ Minimum F1 Score: 1.0

🔢 Classes: 4 Soil Types

🏅 Leaderboard Rank: 41

🤝 Team Members

Amirtha K (amirtha1409)

Tharun Babu C (tharunbabu20)



📎 Acknowledgements

Kaggle - Soil Classification Competition

Annam.ai - IIT Ropar AI Research Center


