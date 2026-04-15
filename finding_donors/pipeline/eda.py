"""
Stage 1 -- Exploratory Data Analysis (EDA)
==========================================
Functions to compute key metrics and generate visualisations
that help understand the census dataset.
"""

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Key Metrics
# ---------------------------------------------------------------------------

def compute_key_metrics(df: pd.DataFrame, target_col: str = "income") -> Dict[str, Any]:
    """
    Compute summary statistics about the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The raw census DataFrame (must contain *target_col*).
    target_col : str
        Name of the income column (default ``"income"``).

    Returns
    -------
    dict
        ``n_records``        -- total number of rows
        ``n_greater_50k``    -- count of individuals earning >50K
        ``n_at_most_50k``    -- count earning <=50K
        ``greater_percent``  -- percentage earning >50K (rounded to 2 dp)
    """
    n_records = len(df)
    n_greater_50k = int((df[target_col] == ">50K").sum())
    n_at_most_50k = int((df[target_col] == "<=50K").sum())
    greater_percent = round(100.0 * n_greater_50k / n_records, 2)

    return {
        "n_records": n_records,
        "n_greater_50k": n_greater_50k,
        "n_at_most_50k": n_at_most_50k,
        "greater_percent": greater_percent,
    }


def print_summary(metrics: Dict[str, Any]) -> None:
    """
    Pretty-print the key metrics dictionary produced by ``compute_key_metrics``.
    """
    print("\n" + "=" * 50)
    print("  KEY METRICS -- Income Distribution")
    print("=" * 50)
    print(f"  Total number of records      : {metrics['n_records']}")
    print(f"  Individuals earning  >50K    : {metrics['n_greater_50k']}")
    print(f"  Individuals earning <=50K    : {metrics['n_at_most_50k']}")
    print(f"  Percentage earning   >50K    : {metrics['greater_percent']}%")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame,
    numerical_features: Optional[List[str]] = None,
    skewed_features: Optional[List[str]] = None,
    target_col: str = "income",
) -> None:
    """
    Generate distribution plots for key features:

    1. Histograms for skewed features (``capital-gain``, ``capital-loss``).
    2. Histograms for other numerical features (age, education-num, hours-per-week).
    3. A bar chart showing the income class balance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw census DataFrame.
    numerical_features : list[str] or None
        Numerical columns to plot. Defaults to
        ``['age', 'education-num', 'hours-per-week']``.
    skewed_features : list[str] or None
        Highly-skewed columns to plot. Defaults to
        ``['capital-gain', 'capital-loss']``.
    target_col : str
        Name of the target column.
    """
    if numerical_features is None:
        numerical_features = ["age", "education-num", "hours-per-week"]
    if skewed_features is None:
        skewed_features = ["capital-gain", "capital-loss"]

    # -- 1. Skewed features (matching visuals.py style) ------------------------
    fig = plt.figure(figsize=(11, 5))
    for i, feature in enumerate(skewed_features):
        ax = fig.add_subplot(1, len(skewed_features), i + 1)
        ax.hist(df[feature], bins=25, color="#00A0A0")
        ax.set_title("'%s' Feature Distribution" % feature, fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])
    fig.suptitle(
        "Skewed Distributions of Continuous Census Data Features",
        fontsize=16, y=1.03,
    )
    fig.tight_layout()
    plt.show()

    # -- 2. Other numerical features -------------------------------------------
    fig = plt.figure(figsize=(11, 5))
    for i, feature in enumerate(numerical_features):
        ax = fig.add_subplot(1, len(numerical_features), i + 1)
        ax.hist(df[feature], bins=25, color="#A000A0")
        ax.set_title("'%s' Feature Distribution" % feature, fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
    fig.suptitle(
        "Distributions of Continuous Census Data Features",
        fontsize=16, y=1.03,
    )
    fig.tight_layout()
    plt.show()

    # -- 3. Target class balance -----------------------------------------------
    counts = df[target_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(counts.index, counts.values, color=["#00A000", "#A00000"],
                  edgecolor="white", width=0.6)
    ax.set_title("Income Class Distribution", fontsize=16)
    ax.set_xlabel("Income", fontsize=12)
    ax.set_ylabel("Number of Records", fontsize=12)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            str(val),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout()
    plt.show()
