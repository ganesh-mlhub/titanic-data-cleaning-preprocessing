"""
=========================================================================
TITANIC DATASET - DATA ACQUISITION, CLEANING & PREPROCESSING PIPELINE
=========================================================================
Single, self-contained script. Run it directly from VS Code (Run > Run
Without Debugging, or the ▶ button) or from a terminal with:

    python titanic_data_cleaning.py

It will automatically:
  1. Create its own data/, images/, and outputs/ folders next to itself
     (so it works no matter what folder your terminal is "in")
  2. Download the raw dataset if it isn't already present
  3. Explore it
  4. Visualize missing values & outliers (before cleaning)
  5. Clean it (missing values, duplicates, inconsistencies)
  6. Treat outliers
  7. Engineer features and preprocess (encode + scale)
  8. Save all cleaned/preprocessed CSVs and charts

Requirements (install once):
    pip install pandas numpy matplotlib seaborn scikit-learn requests
=========================================================================
"""

import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for running outside a GUI / in VS Code terminal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
sns.set_style("whitegrid")

# -------------------------------------------------------------------
# 0. SETUP: paths are anchored to THIS FILE's location, not the
#    terminal's current working directory. This is the #1 reason a
#    script "doesn't work" when opened fresh in VS Code.
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in (DATA_DIR, IMAGES_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_CSV_PATH = DATA_DIR / "titanic_raw.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# -------------------------------------------------------------------
# 1. DATA ACQUISITION
# -------------------------------------------------------------------
banner("STEP 1: DATA ACQUISITION")

if not RAW_CSV_PATH.exists():
    print(f"Dataset not found locally. Downloading from:\n  {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, RAW_CSV_PATH)
        print(f"Downloaded successfully to: {RAW_CSV_PATH}")
    except Exception as e:
        print(f"\nERROR: could not download the dataset automatically ({e}).")
        print("If you're offline or behind a firewall, download it manually from:")
        print(f"  {DATA_URL}")
        print(f"and save it as:\n  {RAW_CSV_PATH}")
        sys.exit(1)
else:
    print(f"Found existing dataset at: {RAW_CSV_PATH}")

df = pd.read_csv(RAW_CSV_PATH)
print(f"\nLoaded dataset with shape: {df.shape}")


# -------------------------------------------------------------------
# 2. INITIAL DATA EXPLORATION
# -------------------------------------------------------------------
banner("STEP 2: INITIAL DATA EXPLORATION")

print("\nFirst 5 rows:\n", df.head())

print("\nData types & non-null counts:")
df.info()

print("\nNumeric summary statistics:\n", df.describe())

print("\nCategorical summary:\n", df.describe(include=['object']))

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_report = missing_report[missing_report['missing_count'] > 0].sort_values(
    'missing_count', ascending=False
)
print("\nMissing values per column:\n", missing_report)

print("\nDuplicate rows:", df.duplicated().sum())

print("\nUnique values in key categorical columns:")
for col in ['Sex', 'Embarked', 'Pclass', 'Survived']:
    print(f"\n{col}:\n{df[col].value_counts(dropna=False)}")

# Save a text log of the exploration step
with open(OUTPUTS_DIR / "step1_exploration_log.txt", "w") as f:
    f.write(f"Shape: {df.shape}\n\n")
    f.write(f"Missing values:\n{missing_report}\n\n")
    f.write(f"Duplicate rows: {df.duplicated().sum()}\n")


# -------------------------------------------------------------------
# 3. VISUALIZE MISSING DATA & OUTLIERS (BEFORE CLEANING)
# -------------------------------------------------------------------
banner("STEP 3: VISUALIZING MISSING VALUES & OUTLIERS (RAW DATA)")

# Missing values heatmap
plt.figure(figsize=(9, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="rocket_r", yticklabels=False)
plt.title("Missing Value Map (raw data)")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "01_missing_heatmap.png", dpi=140)
plt.close()

# Missing values bar chart
missing_nonzero = missing[missing > 0].sort_values(ascending=False)
plt.figure(figsize=(7, 4))
sns.barplot(x=missing_nonzero.values, y=missing_nonzero.index,
            hue=missing_nonzero.index, palette="viridis", legend=False)
plt.xlabel("Number of Missing Values")
plt.title("Missing Values by Column")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "02_missing_bar.png", dpi=140)
plt.close()

# Outlier boxplots (raw)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(x=df['Age'].dropna(), ax=axes[0], color="skyblue")
axes[0].set_title("Age - Boxplot (raw)")
sns.boxplot(x=df['Fare'], ax=axes[1], color="salmon")
axes[1].set_title("Fare - Boxplot (raw)")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "03_outlier_boxplots_raw.png", dpi=140)
plt.close()


def iqr_outlier_count(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper


for col in ['Age', 'Fare']:
    n, low, up = iqr_outlier_count(df[col].dropna())
    pct = n / len(df.dropna(subset=[col])) * 100
    print(f"{col}: {n} outliers outside [{low:.2f}, {up:.2f}] "
          f"using 1.5*IQR rule ({pct:.1f}% of non-null rows)")

print(f"\nSaved raw-data charts to: {IMAGES_DIR}")


# -------------------------------------------------------------------
# 4. CLEANING: DUPLICATES, MISSING VALUES, INCONSISTENCIES
# -------------------------------------------------------------------
banner("STEP 4: CLEANING THE DATA")

print("Starting shape:", df.shape)

# --- 4.1 Duplicates ---
dupes_removed = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicates removed: {dupes_removed}")

# --- 4.2 Embarked: 0.22% missing -> impute with mode ---
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)
print(f"Embarked: filled missing values with mode = '{mode_embarked}'")

# --- 4.3 Age: 19.87% missing -> group-wise median (Pclass x Sex) ---
n_age_missing = df['Age'].isnull().sum()
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
df['Age'] = df['Age'].fillna(df['Age'].median())  # safety fallback
print(f"Age: filled {n_age_missing} missing values using group-wise (Pclass x Sex) median")

# --- 4.4 Cabin: 77% missing -> convert to binary HasCabin, drop raw column ---
df['HasCabin'] = df['Cabin'].notnull().astype(int)
df = df.drop(columns=['Cabin'])
print("Cabin: dropped raw column, replaced with binary 'HasCabin' indicator")

# --- 4.5 Inconsistency checks ---
zero_fare = (df['Fare'] == 0).sum()
print(f"Rows with Fare == 0: {zero_fare} (flagged, not removed - plausible crew/comp tickets)")

df['Sex'] = df['Sex'].str.strip().str.lower()
df['Embarked'] = df['Embarked'].str.strip().str.upper()

assert df['Age'].between(0, 100).all(), "Age out of plausible range"
assert df['Fare'].ge(0).all(), "Negative fare found"
print("Sanity checks passed: Age in [0,100], Fare >= 0")


# -------------------------------------------------------------------
# 5. OUTLIER TREATMENT (Fare)
# -------------------------------------------------------------------
banner("STEP 5: OUTLIER TREATMENT")

q1, q3 = df['Fare'].quantile(0.25), df['Fare'].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
lower_fence = max(0, q1 - 1.5 * iqr)
n_capped = (df['Fare'] > upper_fence).sum()
df['Fare_capped'] = df['Fare'].clip(lower=lower_fence, upper=upper_fence)
print(f"Fare: capped {n_capped} outlier values at upper fence = {upper_fence:.2f} "
      f"(original 'Fare' column retained for traceability)")
print("Age: outliers retained (biologically plausible elderly passengers, not data errors)")


# -------------------------------------------------------------------
# 6. FEATURE ENGINEERING
# -------------------------------------------------------------------
banner("STEP 6: FEATURE ENGINEERING")

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
print("Added features: FamilySize, IsAlone, Title (rare titles grouped)")


# -------------------------------------------------------------------
# 7. ENCODING + SCALING (final preprocessing)
# -------------------------------------------------------------------
banner("STEP 7: ENCODING & SCALING")

df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

scaler = StandardScaler()
num_cols = ['Age', 'Fare_capped', 'FamilySize']
df_encoded[[c + '_scaled' for c in num_cols]] = scaler.fit_transform(df_encoded[num_cols])

df_final = df_encoded.drop(columns=['PassengerId', 'Name', 'Ticket'])

print("Encoded categorical columns: Sex, Embarked, Title (one-hot, drop_first=True)")
print("Scaled numeric columns:", num_cols)
print("Dropped non-predictive columns: PassengerId, Name, Ticket")


# -------------------------------------------------------------------
# 8. SAVE CLEANED & PREPROCESSED DATA
# -------------------------------------------------------------------
banner("STEP 8: SAVING OUTPUTS")

print("Final cleaned shape:", df.shape)
print("Final preprocessed shape:", df_final.shape)
print("Remaining missing values in cleaned df:", df.isnull().sum().sum())
print("Remaining missing values in preprocessed df:", df_final.isnull().sum().sum())

df.to_csv(OUTPUTS_DIR / "titanic_cleaned.csv", index=False)
df_final.to_csv(OUTPUTS_DIR / "titanic_preprocessed.csv", index=False)
print(f"\nSaved:\n  {OUTPUTS_DIR / 'titanic_cleaned.csv'}\n  {OUTPUTS_DIR / 'titanic_preprocessed.csv'}")


# -------------------------------------------------------------------
# 9. AFTER-CLEANING VISUALS (for the report)
# -------------------------------------------------------------------
banner("STEP 9: GENERATING AFTER-CLEANING CHARTS")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(x=df['Fare'], ax=axes[0], color="salmon")
axes[0].set_title("Fare - Before Capping")
sns.boxplot(x=df['Fare_capped'], ax=axes[1], color="lightgreen")
axes[1].set_title("Fare - After Capping (1.5*IQR)")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "04_fare_before_after.png", dpi=140)
plt.close()

plt.figure(figsize=(7, 4))
sns.heatmap(df.isnull(), cbar=False, cmap="rocket_r", yticklabels=False)
plt.title("Missing Value Map AFTER Cleaning (should be empty)")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "05_missing_heatmap_after.png", dpi=140)
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=30, kde=True, color="steelblue")
plt.title("Age Distribution After Imputation")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "06_age_after_imputation.png", dpi=140)
plt.close()

print(f"Saved after-cleaning charts to: {IMAGES_DIR}")

banner("PIPELINE COMPLETE")
print(f"All outputs are in: {OUTPUTS_DIR}")
print(f"All charts are in:  {IMAGES_DIR}")
