# Data Acquisition, Cleaning, and Preprocessing — Titanic Dataset

This repository contains Task 1: **Data Acquisition, Cleaning, and Preprocessing**, completed as part of the Virtual Data Science with Python traineeship.

## Dataset
- **Source:** [Titanic dataset, public GitHub mirror](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) (classic Kaggle "Titanic: Machine Learning from Disaster" dataset)
- **Size:** 891 rows × 12 columns (raw)

## Repository Structure
```
├── titanic_data_cleaning.py     # single script: does EVERYTHING, start to finish
├── requirements.txt             # Python dependencies
├── README.md
├── Data_Cleaning_Report.docx    # full written report
│── .gitignore
│  (created automatically the first time you run the script)
├── data/
│   └── titanic_raw.csv          # downloaded automatically on first run
├── images/                      # charts generated during the run
└── outputs/
    ├── step1_exploration_log.txt
    ├── titanic_cleaned.csv      # cleaned dataset (no missing values, outliers capped)
    └── titanic_preprocessed.csv # encoded + scaled, model-ready dataset
```

## How to Run (VS Code or terminal)

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal inside VS Code (`Terminal > New Terminal`) — this ensures the terminal is in the right folder.
3. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the script:
   ```bash
   python titanic_data_cleaning.py
   ```
   Or in VS Code, just open `titanic_data_cleaning.py` and click the ▶ "Run" button.

The script automatically creates its own `data/`, `images/`, and `outputs/` folders **next to itself** (not wherever your terminal happens to be pointed), downloads the dataset if it isn't already there, and runs the full pipeline end to end. You do not need to create any folders manually.

> **Note:** the script needs an internet connection the first time it runs (to download `titanic_raw.csv`). After that, it reuses the local copy.

## Summary of Work
1. **Exploration** — identified 3 columns with missing data (Cabin 77%, Age 20%, Embarked 0.2%), no duplicates, and outliers in `Fare` (13% of rows via 1.5×IQR rule).
2. **Missing values** — Embarked imputed with mode; Age imputed with group-wise (Pclass × Sex) median; Cabin converted into a binary `HasCabin` feature instead of being imputed directly.
3. **Outliers** — Fare outliers capped (winsorized) at the 1.5×IQR fence rather than deleted, to avoid losing 13% of the data.
4. **Feature engineering** — `FamilySize`, `IsAlone`, and `Title` (extracted from `Name`) added.
5. **Preprocessing** — categorical variables one-hot encoded; numeric features standardized with `StandardScaler`.

Full rationale, code, and discussion of trade-offs is documented in `Data_Cleaning_Report.docx`.
