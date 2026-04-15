"""
Stage 2 -- Data Preprocessing
==============================
Functions to clean, transform, scale, and encode the census data so it is
ready for supervised learning algorithms.

Pipeline steps (in order):
  1. Separate features from target.
  2. Handle missing values (check & fill/drop).
  3. Remove duplicate rows.
  4. Log-transform highly skewed features.
  5. Min-Max scale numerical features.
  6. One-hot encode categorical features.
  7. Encode the binary target variable (income).
"""

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Default column groups (matching the census.csv schema)
# ---------------------------------------------------------------------------
DEFAULT_SKEWED_COLS: List[str] = ["capital-gain", "capital-loss"]

DEFAULT_NUMERICAL_COLS: List[str] = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


# ---------------------------------------------------------------------------
# Data Cleaning (adapted from data_preprocessor/)
# ---------------------------------------------------------------------------

def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table of null counts and null ratios per column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: ``null_count``, ``null_ratio_%``
    """
    null_count = df.isnull().sum()
    null_ratio = (null_count / len(df)) * 100
    return pd.DataFrame({
        "null_count": null_count,
        "null_ratio_%": null_ratio.round(2),
    })


def handle_missing(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_cols: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        Global strategy when *fill_cols* is not provided:
        ``'drop'`` -- drop rows with any null.
        ``'mean'`` / ``'median'`` / ``'mode'`` -- fill all numeric nulls.
    fill_cols : dict or None
        Per-column strategies, e.g. ``{'age': 'median', 'workclass': 'mode'}``.
        Overrides *strategy* when provided.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input.
    """
    cleaned = df.copy()

    if fill_cols:
        for col, strat in fill_cols.items():
            if strat == "mean":
                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
            elif strat == "median":
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            elif strat == "mode":
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
            elif strat == "drop":
                cleaned = cleaned.dropna(subset=[col])
    else:
        if strategy == "drop":
            cleaned = cleaned.dropna()
        elif strategy in ("mean", "median", "mode"):
            num_cols = cleaned.select_dtypes(include=["number"]).columns
            for col in num_cols:
                if strategy == "mean":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
                elif strategy == "median":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                elif strategy == "mode":
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    return cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.

    Returns
    -------
    pd.DataFrame
        De-duplicated copy.
    """
    n_before = len(df)
    cleaned = df.drop_duplicates()
    n_removed = n_before - len(cleaned)
    print(f"[OK] Duplicates removed: {n_removed} rows dropped ({len(cleaned)} remaining)")
    return cleaned





# ---------------------------------------------------------------------------
# Transform / Scale / Encode steps
# ---------------------------------------------------------------------------

def log_transform_skewed(
    features_df: pd.DataFrame,
    skewed_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply a log1p transformation  (``log(x + 1)``) to highly skewed columns.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix (target excluded).
    skewed_cols : list[str] or None
        Columns to transform.  Defaults to ``['capital-gain', 'capital-loss']``.

    Returns
    -------
    pd.DataFrame
        Copy of *features_df* with the specified columns log-transformed.
    """
    if skewed_cols is None:
        skewed_cols = DEFAULT_SKEWED_COLS

    transformed = features_df.copy()
    transformed[skewed_cols] = transformed[skewed_cols].apply(lambda x: np.log(x + 1))
    print(f"[OK] Log-transformed skewed features: {skewed_cols}")
    return transformed


def normalize_numerical(
    features_df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply Min-Max scaling to numerical features (range 0-1).

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix (skewed cols should already be log-transformed).
    numerical_cols : list[str] or None
        Columns to scale.  Defaults to the five standard numerical columns.
    scaler : MinMaxScaler or None
        If provided, ``transform`` is used (useful at inference time).
        If ``None``, a new scaler is ``fit_transform``-ed.

    Returns
    -------
    (pd.DataFrame, MinMaxScaler)
        Scaled feature DataFrame and the fitted scaler.
    """
    if numerical_cols is None:
        numerical_cols = DEFAULT_NUMERICAL_COLS

    scaled = features_df.copy()

    if scaler is None:
        scaler = MinMaxScaler()  # default range (0, 1)
        scaled[numerical_cols] = scaler.fit_transform(scaled[numerical_cols])
        print(f"[OK] Fit & transformed numerical features with MinMaxScaler: {numerical_cols}")
    else:
        scaled[numerical_cols] = scaler.transform(scaled[numerical_cols])
        print(f"[OK] Transformed numerical features using existing scaler: {numerical_cols}")

    return scaled, scaler


def encode_target(income_series: pd.Series) -> pd.Series:
    """
    Convert the raw income label to binary integers.

    ``<=50K`` -> 0 , ``>50K`` -> 1

    Parameters
    ----------
    income_series : pd.Series
        The raw ``income`` column (string values).

    Returns
    -------
    pd.Series
        Integer-encoded target (0 or 1).
    """
    encoded = income_series.apply(lambda x: 1 if x.strip() == ">50K" else 0)
    print(f"[OK] Target encoded: <=50K -> 0, >50K -> 1  |  dtype={encoded.dtype}")
    return encoded


def one_hot_encode(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all remaining categorical (object-type) columns.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with numerical columns already processed.

    Returns
    -------
    pd.DataFrame
        Feature matrix with all categorical columns replaced by dummy variables.
    """
    encoded = pd.get_dummies(features_df)
    print(f"[OK] One-hot encoded categorical features -> {encoded.shape[1]} total features.")
    return encoded


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    target_col: str = "income",
    skewed_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, MinMaxScaler]:
    """
    Run the complete Stage 2 preprocessing pipeline.

    Steps
    -----
    1. Separate features from target.
    2. Handle missing values.
    3. Remove duplicate rows.
    4. Log-transform skewed features.
    5. Min-Max normalise numerical features.
    6. One-hot encode categorical features.
    7. Encode the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The full raw census DataFrame (including the target column).
    target_col : str
        Name of the target column (default ``"income"``).
    skewed_cols : list[str] or None
        Columns to log-transform.
    numerical_cols : list[str] or None
        Columns to scale.

    Returns
    -------
    (features_final, income, scaler)
        features_final : pd.DataFrame - fully processed feature matrix
        income         : pd.Series    - binary-encoded target
        scaler         : MinMaxScaler - fitted scaler (for reuse at inference)
    """
    print("\n" + "=" * 60)
    print("  STAGE 2 -- DATA PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")

    # 1. Separate features / target
    income_raw = df[target_col]
    features_raw = df.drop(target_col, axis=1)
    print(f"  Features shape : {features_raw.shape}")
    print(f"  Target shape   : {income_raw.shape}\n")

    # 2. Handle missing values
    missing_info = check_missing(features_raw)
    total_nulls = int(missing_info["null_count"].sum())
    if total_nulls > 0:
        print(f"  Found {total_nulls} missing values -- filling/dropping...")
        features_raw = handle_missing(features_raw, strategy="drop")
        income_raw = income_raw.loc[features_raw.index]
        print(f"[OK] Missing values handled. Shape now: {features_raw.shape}")
    else:
        print("[OK] No missing values found.")

    # 3. Remove duplicates
    combined = features_raw.copy()
    combined[target_col] = income_raw
    combined = remove_duplicates(combined)
    income_raw = combined[target_col]
    features_raw = combined.drop(target_col, axis=1)

    # 4. Log-transform skewed features
    features_log = log_transform_skewed(features_raw, skewed_cols)

    # 5. Min-Max scale numerical features
    features_scaled, scaler = normalize_numerical(features_log, numerical_cols)

    # 6. One-hot encode categorical features
    features_final = one_hot_encode(features_scaled)

    # 7. Encode target
    income = encode_target(income_raw)

    print(f"\n>> Final features shape : {features_final.shape}")
    print(f">> Final target shape   : {income.shape}")
    print("=" * 60 + "\n")

    return features_final, income, scaler
