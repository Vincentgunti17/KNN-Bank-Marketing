# Week 5 — Supervised Learning with KNN (Portuguese Bank Marketing)

This repository contains a Jupyter notebook that builds and evaluates a K-Nearest Neighbors (KNN) classifier on the Portuguese Bank Marketing dataset. The goal is to predict whether a customer will subscribe to a term deposit based on personal, contact, and socio‑economic attributes.

## Project overview
- Task: Binary classification (subscribe to term deposit: yes or no)
- Algorithm: K-Nearest Neighbors (with preprocessing and grid search)
- Best result observed in the notebook: accuracy ≈ 0.8914 with n_neighbors = 19
- Train/test split: 80/20

## Dataset
The notebook uses the UCI Portuguese Bank Marketing dataset (often named `bank-additional-full.csv`). Place the CSV in a local `data/` folder, or update the path in the notebook cell where the CSV is loaded.

Example expected path in the notebook:
```
data/bank-additional-full.csv
```

If you do not have the file yet, download it from the UCI Machine Learning Repository and save it to `data/`.

## Methods
The workflow in the notebook includes:
1. Importing libraries and loading the dataset
2. Basic exploration and data info (≈ 41,188 rows; mixed numeric and categorical columns)
3. Preprocessing with a `ColumnTransformer`:
   - Numeric pipeline: `SimpleImputer(strategy="median")` + `StandardScaler()`
   - Categorical pipeline: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`
4. End-to-end `Pipeline`: `preprocessor` + `KNeighborsClassifier()`
5. Model selection with `GridSearchCV` over `n_neighbors` from 1 to 20
6. Evaluation on the test set using accuracy, classification report, and confusion matrix

## Results (from the notebook run)
- Best hyperparameter: `n_neighbors = 19`
- Test accuracy: ~0.8914
- Confusion matrix example:
  ```
  [[5138  166]
   [ 496  298]]
   
   <p align="center">
  <img src="images/output.png" alt="Model Results" width="600"/>
</p>

  ```
- The notebook also prints a classification report (precision, recall, f1-score) for both classes.

Note: Your numbers may vary slightly due to environment differences and any changes to preprocessing or random state.

## Repository structure
```
.
├── README.md
├── vincent_w5_Assignment.ipynb
└── data/
    └── bank-additional-full.csv   # not included; add locally
```

## Getting started
1. Clone this repo
2. Create and activate a virtual environment
3. Install dependencies
4. Launch Jupyter and run the notebook

### Quickstart
```bash
git clone <your-repo-url>.git
cd <your-repo-folder>

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (Powershell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

jupyter notebook vincent_w5_Assignment.ipynb
# or
jupyter lab
```

## Requirements
If you do not have a `requirements.txt` yet, start with this minimal set:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

## Reproducibility tips
- Keep `random_state=42` where applicable for consistent splits
- Ensure the CSV path is correct before running the notebook
- If you change columns or encoders, clear outputs and re-run all cells

## Acknowledgments
- Dataset: UCI Machine Learning Repository (Portuguese Bank Marketing)
- Scikit-learn for preprocessing, pipelines, and model selection
