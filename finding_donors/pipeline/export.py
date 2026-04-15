"""
Export -- Save preprocessed data to CSV
========================================
Utility to persist the fully preprocessed features and encoded target
to a CSV file, ready for downstream model training.
"""

import os
import pandas as pd


# Default output sits next to census.csv
_DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "census_preprocessed.csv"
)


def export_preprocessed(
    features: pd.DataFrame,
    income: pd.Series,
    output_path: str = _DEFAULT_OUTPUT,
    include_target: bool = True,
) -> str:
    """
    Save the preprocessed feature matrix (and optionally the target) to CSV.

    Parameters
    ----------
    features : pd.DataFrame
        The fully preprocessed feature matrix (e.g. 103 columns after
        log-transform, scaling, and one-hot encoding).
    income : pd.Series
        Binary-encoded target variable (0 / 1).
    output_path : str
        Destination file path.  Defaults to
        ``finding_donors/census_preprocessed.csv``.
    include_target : bool
        If ``True`` (default), append the ``income`` column to the output.

    Returns
    -------
    str
        The absolute path of the written CSV file.
    """
    df = features.copy()

    if include_target:
        df["income"] = income.values

    df.to_csv(output_path, index=False)

    abs_path = os.path.abspath(output_path)
    print(f"[OK] Preprocessed data exported: {abs_path}")
    print(f"     Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return abs_path
