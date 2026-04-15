"""
Stage 1 -- Data Loading & Initial Inspection
=============================================
Functions to load the census dataset and perform a first-pass inspection.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Default path: census.csv sits one directory above the pipeline/ package.
# ---------------------------------------------------------------------------
_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "census.csv")


def load_data(filepath: str = _DEFAULT_CSV) -> pd.DataFrame:
    """
    Load the census dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.
        Defaults to ``finding_donors/census.csv``.

    Returns
    -------
    pd.DataFrame
        The raw census data with all original columns.
    """
    df = pd.read_csv(filepath)
    # Strip any leading/trailing whitespace from string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    print(f"[OK] Data loaded successfully: {df.shape[0]} records, {df.shape[1]} columns.")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a concise summary of the dataset.

    Displays:
    - First 5 rows
    - Shape (records x features)
    - Data types
    - Null counts per column
    - Basic descriptive statistics for numerical features

    Parameters
    ----------
    df : pd.DataFrame
        The raw census DataFrame.
    """
    print("\n" + "=" * 60)
    print("  DATASET INSPECTION")
    print("=" * 60)

    print(f"\n  Shape: {df.shape[0]} records x {df.shape[1]} columns\n")

    print("-- First 5 rows --")
    print(df.head().to_string())

    print("\n-- Column Data Types --")
    print(df.dtypes.to_string())

    print("\n-- Null / Missing Values --")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("  No missing values found. [OK]")
    else:
        print(null_counts[null_counts > 0].to_string())

    print("\n-- Descriptive Statistics (numerical) --")
    print(df.describe().to_string())

    print("\n" + "=" * 60 + "\n")
