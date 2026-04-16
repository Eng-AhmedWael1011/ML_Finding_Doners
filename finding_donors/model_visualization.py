"""
Model Visualization Module
===========================
Visualization utilities and plotting functions for Finding Donors analysis.

Provides:
  - Confusion matrix heatmaps
  - ROC curves
  - Feature importance bar plots
  - Comparison charts
  - Learning curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List

from sklearn.metrics import confusion_matrix, roc_curve, auc


# Set default style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


class ModelVisualizer:
    """Visualization utilities for model results."""
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        ax: Optional[plt.Axes] = None,
        normalize: bool = False,
    ) -> plt.Axes:
        """
        Plot confusion matrix as heatmap.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        title : str
            Plot title.
        ax : matplotlib Axes or None
            Subplot axis. If None, creates new figure.
        normalize : bool
            If True, normalize rows to percentages.
        
        Returns
        -------
        plt.Axes
            The axis object.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2%"
        else:
            fmt = "d"
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            cbar=True,
            ax=ax,
            xticklabels=["<=50K", ">50K"],
            yticklabels=["<=50K", ">50K"],
        )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        
        return ax
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot ROC curve.
        
        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_proba : array-like
            Predicted probabilities (positive class).
        title : str
            Plot title.
        ax : matplotlib Axes or None
        
        Returns
        -------
        plt.Axes
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random Classifier")
        
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_feature_importance(
        importance_df: pd.DataFrame,
        top_n: int = 15,
        title: str = "Feature Importance",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot feature importance as horizontal bar chart.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with columns 'Feature' and 'Importance' (or 'Importance %').
        top_n : int
            Number of top features to display.
        title : str
            Plot title.
        ax : matplotlib Axes or None
        
        Returns
        -------
        plt.Axes
        """
        # Determine importance column
        importance_col = None
        for col in ["Importance %", "Importance", "Mean_SHAP"]:
            if col in importance_df.columns:
                importance_col = col
                break
        
        if importance_col is None:
            raise ValueError("No importance column found in DataFrame")
        
        plot_data = importance_df.head(top_n).sort_values(importance_col)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(plot_data["Feature"], plot_data[importance_col], color="steelblue")
        
        ax.set_xlabel(importance_col, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_model_comparison(
        comparison_df: pd.DataFrame,
        metric: str = "Test F0.5",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot comparison of multiple models.
        
        Parameters
        ----------
        comparison_df : pd.DataFrame
            DataFrame with classifier names and metrics.
        metric : str
            Metric column to plot.
        ax : matplotlib Axes or None
        
        Returns
        -------
        plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in comparison DataFrame")
        
        plot_data = comparison_df.sort_values(metric)
        
        ax.barh(plot_data["Classifier"], plot_data[metric], color="forestgreen")
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f"Model Comparison: {metric}", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_learning_curve(
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        title: str = "Learning Curve",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot learning curve (training vs test score vs training set size).
        
        Parameters
        ----------
        train_sizes : array
            Training set sizes.
        train_scores : array
            Training scores (mean across CV folds).
        test_scores : array
            Validation scores (mean across CV folds).
        title : str
            Plot title.
        ax : matplotlib Axes or None
        
        Returns
        -------
        plt.Axes
        """
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_mean, "o-", label="Training Score", lw=2)
        ax.fill_between(
            train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1
        )
        
        ax.plot(train_sizes, test_mean, "o-", label="Cross-Validation Score", lw=2)
        ax.fill_between(
            train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1
        )
        
        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    @staticmethod
    def plot_metrics_comparison(
        metrics_dict: Dict[str, Dict[str, float]],
        metrics_to_plot: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot comparison of multiple metrics across datasets.
        
        Parameters
        ----------
        metrics_dict : dict
            Nested dict {dataset_name: {metric_name: value}}.
        metrics_to_plot : list[str] or None
            Which metrics to include. If None, uses common metrics.
        ax : matplotlib Axes or None
        
        Returns
        -------
        plt.Axes
        """
        if metrics_to_plot is None:
            metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
        
        df = pd.DataFrame(metrics_dict).T
        
        # Filter to available metrics
        available = [m for m in metrics_to_plot if m in df.columns]
        plot_data = df[available]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_data.plot(kind="bar", ax=ax, width=0.8)
        
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Metrics Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        return ax
    
    @staticmethod
    def plot_partial_dependence(
        x_values: np.ndarray,
        y_values: np.ndarray,
        feature_name: str,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot partial dependence for a single feature.
        
        Parameters
        ----------
        x_values : array
            Feature values.
        y_values : array
            Average predictions.
        feature_name : str
            Name of the feature.
        ax : matplotlib Axes or None
        
        Returns
        -------
        plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x_values, y_values, lw=2, color="steelblue")
        ax.fill_between(x_values, y_values, alpha=0.3, color="steelblue")
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Average Prediction", fontsize=12)
        ax.set_title(f"Partial Dependence: {feature_name}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        return ax


def create_evaluation_dashboard(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    y_proba_test: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Create a 2x2 dashboard with key evaluation plots.
    
    Parameters
    ----------
    y_train, y_pred_train : training data
    y_test, y_pred_test : test data
    y_proba_test : predicted probabilities (optional, for ROC curve)
    
    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion matrices
    ModelVisualizer.plot_confusion_matrix(y_train, y_pred_train, "Train Confusion Matrix", axes[0, 0])
    ModelVisualizer.plot_confusion_matrix(y_test, y_pred_test, "Test Confusion Matrix", axes[0, 1])
    
    # ROC curve (if probabilities available)
    if y_proba_test is not None:
        ModelVisualizer.plot_roc_curve(y_test, y_proba_test, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, "Probabilities not available", ha="center", va="center")
        axes[1, 0].set_title("ROC Curve (N/A)")
    
    # Metrics summary
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics_text = f"""
    TRAINING SET:
    Accuracy: {accuracy_score(y_train, y_pred_train):.4f}
    Precision: {precision_score(y_train, y_pred_train):.4f}
    Recall: {recall_score(y_train, y_pred_train):.4f}
    F1-Score: {f1_score(y_train, y_pred_train):.4f}
    
    TEST SET:
    Accuracy: {accuracy_score(y_test, y_pred_test):.4f}
    Precision: {precision_score(y_test, y_pred_test):.4f}
    Recall: {recall_score(y_test, y_pred_test):.4f}
    F1-Score: {f1_score(y_test, y_pred_test):.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontfamily="monospace", fontsize=11, verticalalignment="center")
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Performance Metrics")
    
    plt.tight_layout()
    return fig
