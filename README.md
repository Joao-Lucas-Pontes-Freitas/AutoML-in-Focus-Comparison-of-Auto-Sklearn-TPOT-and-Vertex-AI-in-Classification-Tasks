# AutoML in Focus: Comparison of Auto-Sklearn, TPOT, and Vertex AI in Classification Tasks

This repository contains the code and datasets used in the experiments of the work *AutoML in Focus: Comparison of Auto-Sklearn, TPOT, and Vertex AI in Classification Tasks*.  
The project evaluates the performance of three AutoML tools (Auto-Sklearn, TPOT, and Vertex AI) on datasets of different modalities (tabular, image, and text), under both clean and noisy conditions.

## Repository Structure

```
.
├── data/              # Original and noisy datasets (tabular, image, text)
│   └── noise/         # Datasets with label, feature, and combined noise
├── notebooks/
│   ├── experiments_Auto-Sklearn.ipynb   # Experiments with Auto-Sklearn
│   ├── experiments_TPOT.ipynb           # Experiments with TPOT
│   └── noise/                           # Scripts to generate noisy datasets
├── src/
│   ├── metrics.py     # Functions to compute evaluation metrics
│   ├── noise.py       # Functions to inject label/feature noise
│   └── serialize.py   # Utilities to serialize and load models
└── README.md
```

## Datasets
The repository includes training and testing sets for multiple datasets:
- **Tabular:** Academic Success, Breast Cancer, Spambase, Wine, Iris  
- **Image:** Digits, MNIST  
- **Text:** IMDB reviews  

Noisy versions of each dataset are provided under `data/noise/`, covering:
- *label noise* (flipped labels)  
- *feature noise* (perturbed attributes, image noise, token removal)  
- *combined noise* (label + feature)

## Notebooks
- **experiments_datasets.ipynb** – Runs experiments with Auto-Sklearn.  
- **experiments_TPOT.ipynb** – Runs experiments with TPOT.  
- **noise/\*** – Generate noisy versions of datasets for evaluation.  

## Source Code
- **metrics.py** – Accuracy, precision, recall, F1, confusion matrix, learning curves.  
- **noise.py** – Functions to inject label and feature noise (tabular, image, text).  
- **serialize.py** – Serialization of models (base64 or GridFS).  