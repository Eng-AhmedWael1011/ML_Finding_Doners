"""
Run Pipeline -- Stages 1 & 2
=============================
End-to-end runner that executes:
  - Stage 1: Data loading, inspection, EDA metrics & visualisations
  - Stage 2: Full preprocessing pipeline

Usage (from the ``finding_donors/`` directory):
    python -m pipeline.run_pipeline

Or import and call ``run()`` programmatically.
"""

from .data_loader import load_data, inspect_data
from .eda import compute_key_metrics, print_summary, plot_feature_distributions
from .preprocessing import preprocess
from .export import export_preprocessed


def run(csv_path: str = None, show_plots: bool = True, export_csv: bool = True, output_path: str = None):
    """
    Execute Stages 1 & 2 of the Finding Donors pipeline.

    Parameters
    ----------
    csv_path : str or None
        Path to the census CSV file.  None uses the default path.
    show_plots : bool
        Whether to display matplotlib plots (disable for CI/testing).
    export_csv : bool
        Whether to export the preprocessed data to CSV (default True).
    output_path : str or None
        Custom path for the exported CSV. None uses the default.

    Returns
    -------
    dict
        ``features_final`` : pd.DataFrame
        ``income``          : pd.Series
        ``scaler``          : sklearn MinMaxScaler (fitted)
        ``metrics``         : dict of EDA key metrics
    """
    # -- Stage 1: Data Loading -------------------------------------------------
    print("\n" + "=" * 60)
    print("  STAGE 1 -- DATA EXPLORATION (EDA)")
    print("=" * 60 + "\n")

    if csv_path:
        df = load_data(csv_path)
    else:
        df = load_data()

    inspect_data(df)

    # Key metrics
    metrics = compute_key_metrics(df)
    print_summary(metrics)

    # Visualisations
    if show_plots:
        plot_feature_distributions(df)

    # -- Stage 2: Preprocessing ------------------------------------------------
    features_final, income, scaler = preprocess(df)

    # -- Export preprocessed data ----------------------------------------------
    exported_path = None
    if export_csv:
        if output_path:
            exported_path = export_preprocessed(features_final, income, output_path)
        else:
            exported_path = export_preprocessed(features_final, income)

    # Final summary
    print(">> Pipeline complete!")
    print(f"   - Features matrix : {features_final.shape[0]} samples x {features_final.shape[1]} features")
    print(f"   - Target vector   : {income.shape[0]} samples")
    print(f"   - Income=1 count  : {income.sum()}  ({round(100.0 * income.sum() / len(income), 2)}%)")
    if exported_path:
        print(f"   - Exported CSV    : {exported_path}")
    print()

    return {
        "features_final": features_final,
        "income": income,
        "scaler": scaler,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Allow running as:  python -m pipeline.run_pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
