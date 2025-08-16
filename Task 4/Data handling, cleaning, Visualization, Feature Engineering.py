import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, PolynomialFeatures
)
DATASET_PATH = "KaggleV2-May-2016.csv"
BASE_DIR = os.path.dirname(DATASET_PATH)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figs")
PLOTLY_DIR = os.path.join(OUT_DIR, "plotly")

for d in [OUT_DIR, FIG_DIR, PLOTLY_DIR]:
    os.makedirs(d, exist_ok=True)


# Data Handling (NumPy)

print("\n NumPy Handling Demo ")
arr = np.arange(1, 13).reshape(3, 4)
print("Array:\n", arr)
print("Shape:", arr.shape, "| Mean:", arr.mean(), "| Col means:", arr.mean(axis=0))
print("Boolean mask arr > 5:\n", arr > 5)
print("Vectorized add 10:\n", arr + 10)


# Load dataset (Pandas)

print("\n Loading Dataset ")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Could not find: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(3))


# Data Handling (Pandas)

print("\n Pandas Handling Demo ")
print("Info:")
df.info()
print("\nDescribe (numeric):")
print(df.describe())
print("\nMissing values per column:\n", df.isna().sum())

# Quick selections
num_cols_initial = df.select_dtypes(include=np.number).columns.tolist()
cat_cols_initial = df.select_dtypes(exclude=np.number).columns.tolist()
print("\nNumeric columns:", num_cols_initial[:10], "...")
print("Categorical columns:", cat_cols_initial[:10], "...")


# 3) Cleaning

print("\n Cleaning ")
# Standardize column names
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Fix known misspellings for this dataset
rename_map = {
    "Hipertension": "Hypertension",
    "Handcap": "Handicap",
    "No-show": "No_show"  # keep underscore for ease
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# Strip string columns
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

# Type conversions (dates)
for c in ["ScheduledDay", "AppointmentDay"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
print(f"Dropped duplicates: {before - df.shape[0]}")

# Fix invalid ages
if "Age" in df.columns:
    neg = (df["Age"] < 0).sum()
    if neg:
        print(f"Fixing {neg} negative ages to NaN.")
    df.loc[df["Age"] < 0, "Age"] = np.nan
    # Clip to plausible range
    df["Age"] = df["Age"].clip(lower=0, upper=110)

# Simple missing handling:
for c in df.columns:
    if df[c].dtype == "O":
        mode_val = df[c].mode(dropna=True)
        fill = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        df[c] = df[c].fillna(fill)
    else:
        df[c] = df[c].fillna(df[c].median())

# Optional outlier capping on numeric via IQR
def iqr_clip(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return s.clip(lower=low, upper=high)

for c in df.select_dtypes(include=np.number).columns:
    df[c] = iqr_clip(df[c])

print("Cleaning complete.")


# 4) Feature Engineering 

print("\n Feature Engineering ")
# Task-specific datetime features
if {"ScheduledDay", "AppointmentDay"}.issubset(df.columns):
    df["Waiting_Days"] = (df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()).dt.days
    df["Waiting_Days"] = df["Waiting_Days"].clip(lower=0)

for c in ["ScheduledDay", "AppointmentDay"]:
    if c in df.columns:
        df[f"{c}_Year"] = df[c].dt.year
        df[f"{c}_Month"] = df[c].dt.month
        df[f"{c}_Day"] = df[c].dt.day
        df[f"{c}_DOW"] = df[c].dt.dayofweek
if "AppointmentDay_DOW" in df.columns:
    df["Is_Weekend"] = df["AppointmentDay_DOW"].isin([5, 6]).astype(int)

# Target encoding (1 for Yes and 0 for No)
target_col = None
for c in df.columns:
    if c.lower().replace("-", "").replace("_", "") == "noshow" or c == "No_show":
        target_col = c
        break
if target_col and df[target_col].dtype == "O":
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0}).astype(float)

# Binary/label encodings (examples)
if "Gender" in df.columns:
    # Female=1, Male=0
    df["is_female"] = (df["Gender"].astype(str).str.upper().str[0] == "F").astype(int)

# Age binning
if "Age" in df.columns:
    df["Age_Bin"] = pd.cut(
        df["Age"],
        bins=[0, 5, 12, 18, 30, 45, 60, 75, 110],
        labels=["0-5", "6-12", "13-18", "19-30", "31-45", "46-60", "61-75", "76+"],
        include_lowest=True
    )

# Frequency encoding (example: Neighbourhood)
if "Neighbourhood" in df.columns:
    freq = df["Neighbourhood"].value_counts()
    df["Neighbourhood_freq"] = df["Neighbourhood"].map(freq).astype(float)

# Target (mean) 
if target_col and "Neighbourhood" in df.columns:
    means = df.groupby("Neighbourhood")[target_col].mean()
    df["Neighbourhood_target_mean"] = df["Neighbourhood"].map(means).astype(float)

# Patient-level aggregates
if "PatientId" in df.columns:
    ag = (df.groupby("PatientId")
            .agg(
                patient_total_visits=("PatientId", "count"),
                patient_avg_age=("Age", "mean"),
                patient_sms_sum=("SMS_received", "sum") if "SMS_received" in df.columns else ("PatientId", "count")
            )
            .reset_index())
    df = df.merge(ag, on="PatientId", how="left")

# One-Hot encoding (safe limited set)
ohe_cols = []
for c in ["Neighbourhood", "Age_Bin"]:
    if c in df.columns and str(df[c].dtype) in ["object", "category"]:
        ohe_cols.append(c)

df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# Scaling 
sc_minmax = MinMaxScaler()
sc_std = StandardScaler()
scale_candidates = ["Age", "Waiting_Days"]
scale_candidates = [c for c in scale_candidates if c in df.columns]
if scale_candidates:
    df[[f"{c}_minmax" for c in scale_candidates]] = sc_minmax.fit_transform(df[scale_candidates])
    df[[f"{c}_std" for c in scale_candidates]] = sc_std.fit_transform(df[scale_candidates])

# Log transform 
for c in ["Waiting_Days"]:
    if c in df.columns and (df[c] > 0).any():
        df[f"{c}_log1p"] = np.log1p(df[c].clip(lower=0))

# Interaction & polynomial features 
numeric_small = [c for c in ["Age", "Waiting_Days"] if c in df.columns]
if numeric_small:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_data = poly.fit_transform(df[numeric_small])
    poly_cols = poly.get_feature_names_out(numeric_small)
    poly_df = pd.DataFrame(poly_data, columns=[f"poly_{c}" for c in poly_cols], index=df.index)
    df = pd.concat([df, poly_df], axis=1)

print("Feature engineering complete. Current shape:", df.shape)


# 5) Manipulation (grouping, pivoting, reshaping)

print("\n Manipulation Examples ")
# GroupBy examples
if "AppointmentDay_DOW" in df.columns and target_col:
    dow_grp = df.groupby("AppointmentDay_DOW")[target_col].agg(["mean", "count"]).reset_index()
    print("No-show rate by day-of-week:\n", dow_grp.head())

# Pivot table
if all(c in df.columns for c in ["AppointmentDay_DOW", "is_female"]) and target_col:
    pv = pd.pivot_table(df, values=target_col, index="AppointmentDay_DOW", columns="is_female", aggfunc="mean")
    print("\nPivot (No-show mean by DOW x Gender):\n", pv)

# Melt
if "Age_Bin" in df.columns and "Age" in df.columns:
    melted = pd.melt(df[["Age_Bin", "Age"]].head(10), id_vars="Age_Bin", var_name="metric", value_name="value")
    print("\nMelt demo (first 10 rows):\n", melted.head())


# 6) Visualizations (Matplotlib, Seaborn, Plotly)

print("\n Visualizations ")
sns.set_theme(style="whitegrid")

def saveplt(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved:", path)

# (A) Matplotlib: histograms, boxplots, bar, line
if "Age" in df.columns:
    plt.figure(figsize=(7,4))
    df["Age"].plot(kind="hist", bins=30, edgecolor="black")
    plt.title("Age Histogram (Matplotlib)")
    saveplt("matplotlib_hist_age")

if "Waiting_Days" in df.columns:
    plt.figure(figsize=(7,4))
    df["Waiting_Days"].plot(kind="box")
    plt.title("Waiting Days Boxplot (Matplotlib)")
    saveplt("matplotlib_box_waiting_days")

if target_col:
    plt.figure(figsize=(6,4))
    df[target_col].value_counts().sort_index().plot(kind="bar")
    plt.title("Target Distribution (Matplotlib)")
    saveplt("matplotlib_bar_target")

# (B) Seaborn: countplot, boxplot, heatmap, pairplot
if target_col:
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df)
    plt.title("No-show Count (Seaborn)")
    saveplt("seaborn_count_target")

if target_col and "Age" in df.columns:
    plt.figure(figsize=(7,4))
    sns.boxplot(x=target_col, y="Age", data=df)
    plt.title("Age by No-show (Seaborn)")
    saveplt("seaborn_box_age_by_target")

# Correlation heatmap (numeric)
num_df = df.select_dtypes(include=np.number)
if num_df.shape[1] >= 2:
    plt.figure(figsize=(10,7))
    sns.heatmap(num_df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (Seaborn)")
    saveplt("seaborn_corr_heatmap")

# Pairplot on a safe subset
pair_cols = [c for c in ["Age", "Waiting_Days", "is_female"] if c in df.columns]
if target_col and len(pair_cols) >= 2:
    sns.pairplot(df.sample(min(1000, len(df))), vars=pair_cols, hue=target_col, corner=True)
    plt.suptitle("Pairplot (subset)", y=1.02)
    saveplt("seaborn_pairplot_subset")

# (C) Plotly: interactive histogram, scatter, box, violin
def save_plotly(fig, name):
    out = os.path.join(PLOTLY_DIR, f"{name}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print("Saved:", out)

if target_col and "Age" in df.columns:
    fig = px.histogram(df, x="Age", color=target_col, nbins=30, title="Age Distribution by No-show (Plotly)", marginal="box")
    save_plotly(fig, "plotly_hist_age_by_target")

if target_col and "Waiting_Days" in df.columns and "Age" in df.columns:
    fig = px.scatter(df, x="Waiting_Days", y="Age", color=target_col, title="Waiting Days vs Age by No-show (Plotly)")
    save_plotly(fig, "plotly_scatter_waiting_vs_age")

if "Neighbourhood" in df.columns:
    neigh_counts = df["Neighbourhood"].value_counts().reset_index()
    neigh_counts.columns = ["Neighbourhood", "count"]
    fig = px.bar(neigh_counts, x="Neighbourhood", y="count", title="Appointments by Neighbourhood (Plotly)")
    fig.update_layout(xaxis_tickangle=-70)
    save_plotly(fig, "plotly_bar_neighbourhood")

# Violin for Age by target
if target_col and "Age" in df.columns:
    fig = px.violin(df, y="Age", x=target_col, box=True, points="outliers", title="Age vs No-show (Violin, Plotly)")
    save_plotly(fig, "plotly_violin_age_by_target")

print("Visualization export complete.")


# 7) Save outputs

clean_path = os.path.join(OUT_DIR, "cleaned_kaggle_dataset.csv")
df.to_csv(clean_path, index=False)
print("\nSaved cleaned dataset:", clean_path)

engineered = df.copy()
small_cat_cols = []
for c in ["Gender"]:
    if c in engineered.columns and str(engineered[c].dtype) == "object" and engineered[c].nunique() <= 6:
        small_cat_cols.append(c)
if small_cat_cols:
    engineered = pd.get_dummies(engineered, columns=small_cat_cols, drop_first=True)

eng_path = os.path.join(OUT_DIR, "engineered_kaggle_dataset.csv")
engineered.to_csv(eng_path, index=False)
print("Saved engineered dataset:", eng_path)
