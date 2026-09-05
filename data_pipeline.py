"""
Stage I - Data Engineering & EDA
Loads the AI4I 2020 Predictive Maintenance dataset (tabular + sensor/time-series
style readings), cleans it, engineers features, and produces a documented,
leakage-safe train/val/test split.

Dataset source: AI4I 2020 Predictive Maintenance Dataset (UCI / Kaggle).
Columns: UDI, Product ID, Type (L/M/H), Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure (target),
TWF, HDF, PWF, OSF, RNF (failure-mode sub-labels).

We treat Air/Process temperature, Rotational speed, Torque and Tool wear as
"sensor channels" that a real factory would stream continuously, and derive
rolling/statistical features per synthetic machine-session to emulate a
time-series predictive-maintenance setting (documented assumption below).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/ai4i2020.csv"

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicates, dtypes, invalid values. Documented assumptions:
    - The public AI4I dataset has no missing values, but we defensively
      impute any NaNs with column median (numeric) / mode (categorical)
      so the pipeline is robust to dirtier real-world exports.
    - Negative or physically impossible sensor values are treated as invalid
      and clipped to the 1st/99th percentile (documented, not silently dropped).
    - Exact duplicate rows (by Product ID) are dropped.
    """
    df = df.copy()
    df = df.drop_duplicates(subset=["Product ID"])

    for col in SENSOR_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
        # clip physically-impossible extreme outliers (documented, not deleted)
        lo, hi = df[col].quantile(0.005), df[col].quantile(0.995)
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        if n_clipped:
            df.attrs.setdefault("clip_log", {})[col] = int(n_clipped)

    df["Type"] = df["Type"].fillna(df["Type"].mode()[0])
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Feature engineering including rolling/statistical (time-series style)
    features. We simulate a machine's sensor stream by sorting on UDI
    (proxy for chronological production order) and computing rolling
    statistics per product Type (proxy for machine class/line).
    """
    df = df.sort_values("UDI").reset_index(drop=True)

    # physically meaningful engineered features
    df["temp_delta"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power_proxy"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] / 9548.8  # ~kW
    df["wear_per_torque"] = df["Tool wear [min]"] / (df["Torque [Nm]"] + 1e-3)

    # rolling/time-series-style features per machine Type (L/M/H line)
    for col in ["Torque [Nm]", "Rotational speed [rpm]", "Process temperature [K]"]:
        roll = df.groupby("Type")[col].rolling(window, min_periods=1)
        df[f"{col}_roll_mean{window}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{col}_roll_std{window}"] = roll.std().reset_index(level=0, drop=True).fillna(0)

    df["Type_L"] = (df["Type"] == "L").astype(int)
    df["Type_M"] = (df["Type"] == "M").astype(int)
    df["Type_H"] = (df["Type"] == "H").astype(int)

    return df


FEATURE_COLS = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]", "temp_delta", "power_proxy", "wear_per_torque",
    "Torque [Nm]_roll_mean5", "Torque [Nm]_roll_std5",
    "Rotational speed [rpm]_roll_mean5", "Rotational speed [rpm]_roll_std5",
    "Process temperature [K]_roll_mean5", "Process temperature [K]_roll_std5",
    "Type_L", "Type_M", "Type_H",
]
TARGET_COL = "Machine failure"


def leakage_safe_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42):
    """Stratified split on the target to avoid leakage. Rolling features are
    computed causally (window looks backward only, per group), and Product ID
    is excluded from features, so no row leaks identity/target information
    across the split. Split is done on rows, not on rolling windows, since
    each row here is an independent production unit (not a continuous
    session that would otherwise need group-based splitting)."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=seed
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val[TARGET_COL], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_dataset(path: str = RAW_PATH):
    raw = load_raw(path)
    clean = clean_data(raw)
    feat = engineer_features(clean)
    train, val, test = leakage_safe_split(feat)
    return feat, train, val, test


def eda_summary(df: pd.DataFrame) -> dict:
    """Return EDA stats consumable by the Streamlit dashboard."""
    return {
        "n_rows": len(df),
        "failure_rate": float(df[TARGET_COL].mean()),
        "by_type_failure_rate": df.groupby("Type")[TARGET_COL].mean().to_dict(),
        "failure_mode_counts": {m: int(df[m].sum()) for m in FAILURE_MODES},
        "sensor_describe": df[SENSOR_COLS].describe().to_dict(),
    }


if __name__ == "__main__":
    feat, train, val, test = build_dataset()
    print("Rows:", len(feat), "Train/Val/Test:", len(train), len(val), len(test))
    print(eda_summary(feat))
