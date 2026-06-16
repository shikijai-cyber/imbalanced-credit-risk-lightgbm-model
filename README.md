# Imbalanced Credit Risk Classification with LightGBM

## Overview
This machine learning project tackles an imbalanced binary classification problem for credit-risk prediction. The workflow compares model strategies and selects a LightGBM-based champion model with undersampling and feature-pruning decisions.

## Features
- Exploratory data analysis for a high-dimensional tabular dataset
- Missing-value handling and feature preparation
- Baseline and tree-based model comparison
- Random undersampling for class imbalance
- XGBoost and LightGBM experiments
- ROC-AUC, precision-recall, confusion-matrix, and classification-report evaluation
- Technical report and pitch deck included

## Project structure
```text
notebooks/     Final modelling notebook
docs/          Technical report and presentation
data/raw/      Source CSV placeholder
requirements.txt
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Place the source file in `data/raw/`:
```text
datafile_full.csv
```

Then run:
```bash
jupyter notebook notebooks/lightgbm_credit_default_model.ipynb
```

## Technologies used
Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, imbalanced-learn, Matplotlib, seaborn.

## Portfolio note
The source CSV was not included in the uploaded archive. The repository is structured to accept the dataset locally without committing large or private data.
