"""
Finding Donors Pipeline Package
================================
Modular pipeline for the CharityML Finding Donors project.

Stage 1: Data Exploration (EDA)
Stage 2: Data Preprocessing
"""

from .data_loader import load_data, inspect_data
from .eda import compute_key_metrics, print_summary, plot_feature_distributions
from .preprocessing import (
    check_missing,
    handle_missing,
    remove_duplicates,
    log_transform_skewed,
    normalize_numerical,
    encode_target,
    one_hot_encode,
    preprocess,
)
from .export import export_preprocessed

__all__ = [
    # Stage 1 - Data Loading
    "load_data",
    "inspect_data",
    # Stage 1 - EDA
    "compute_key_metrics",
    "print_summary",
    "plot_feature_distributions",
    # Stage 2 - Data Cleaning
    "check_missing",
    "handle_missing",
    "remove_duplicates",
    # Stage 2 - Transform / Scale / Encode
    "log_transform_skewed",
    "normalize_numerical",
    "encode_target",
    "one_hot_encode",
    "preprocess",
    # Export
    "export_preprocessed",
]
