"""
Model Interpretation Module
============================
Feature importance analysis, model explanation, and interpretation for Finding Donors.

Provides:
  - Feature importance extraction and ranking
  - Permutation importance
  - SHAP value explanations (if available)
  - Partial dependence plots
  - Model-agnostic explanations
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis for ClassificationModels.
    
    Supports:
      - Tree-based feature importance (.feature_importances_)
      - Permutation importance (model-agnostic)
      - SHAP values (if installed)
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        model : sklearn estimator
            Fitted classifier with prediction capability.
        feature_names : list[str] or None
            Column names for features. If None, uses generic names (Feature_0, Feature_1, ...).
        """
        self.model = model
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(100)]
        self.importance_scores = {}
    
    def get_tree_importance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Works with: RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, etc.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (used to get feature count).
        
        Returns
        -------
        pd.DataFrame
            Features ranked by importance (Feature, Importance, Rank).
        """
        if not hasattr(self.model, "feature_importances_"):
            warnings.warn(f"Model {self.model.__class__.__name__} has no feature_importances_ attribute.")
            return pd.DataFrame()
        
        importances = self.model.feature_importances_
        feature_names = X_train.columns[:len(importances)]
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        
        importance_df["Rank"] = importance_df.index + 1
        importance_df["Importance %"] = (importance_df["Importance"] * 100).round(2)
        
        self.importance_scores["tree_importance"] = importance_df
        return importance_df
    
    def get_permutation_importance(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Calculate permutation feature importance (model-agnostic).
        
        Measures drop in model performance when each feature is randomly shuffled.
        Works with any model.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : array-like
            Test labels.
        n_repeats : int
            Number of times to permute each feature.
        random_state : int
            Random seed.
        
        Returns
        -------
        pd.DataFrame
            Features ranked by permutation importance.
        """
        perm_importance = permutation_importance(
            self.model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        
        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": perm_importance.importances_mean,
            "Std": perm_importance.importances_std,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        
        importance_df["Rank"] = importance_df.index + 1
        importance_df["Importance %"] = (importance_df["Importance"] * 100).round(2)
        
        self.importance_scores["permutation_importance"] = importance_df
        return importance_df
    
    def get_shap_importance(
        self,
        X_train: pd.DataFrame,
        sample_size: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Calculate SHAP feature importance (requires shap package).
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (used to create background).
        sample_size : int or None
            Sample size for background data (for speed). If None, uses all.
        
        Returns
        -------
        pd.DataFrame or None
            Features ranked by mean absolute SHAP value.
            Returns None if SHAP is not available or calculation fails.
        """
        try:
            import shap
        except ImportError:
            warnings.warn("SHAP not installed. Install with: pip install shap")
            return None
        
        try:
            # Create explainer
            background = X_train if sample_size is None else X_train.sample(sample_size, random_state=42)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_train)
            
            # Handle multiclass/binary output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary, use positive class
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            importance_df = pd.DataFrame({
                "Feature": X_train.columns,
                "Mean_SHAP": mean_abs_shap,
            }).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
            
            importance_df["Rank"] = importance_df.index + 1
            importance_df["SHAP %"] = (importance_df["Mean_SHAP"] * 100).round(2)
            
            self.importance_scores["shap_importance"] = importance_df
            return importance_df
        
        except Exception as e:
            warnings.warn(f"SHAP calculation failed: {e}")
            return None
    
    def get_top_features(
        self,
        importance_type: str = "tree",
        top_n: int = 5,
    ) -> List[str]:
        """
        Get top N most important features.
        
        Parameters
        ----------
        importance_type : str
            'tree', 'permutation', or 'shap'.
        top_n : int
            Number of top features to return.
        
        Returns
        -------
        list[str]
            Top N feature names.
        """
        key_map = {
            "tree": "tree_importance",
            "permutation": "permutation_importance",
            "shap": "shap_importance",
        }
        
        key = key_map.get(importance_type)
        if key not in self.importance_scores:
            return []
        
        df = self.importance_scores[key]
        return df.head(top_n)["Feature"].tolist()
    
    def get_partial_dependence(
        self,
        X_train: pd.DataFrame,
        feature_name: str,
        percentiles: Tuple[float, float] = (0.05, 0.95),
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate partial dependence for a single feature.
        
        Shows average model prediction as a function of feature values.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training data (for context).
        feature_name : str
            Name of the feature to compute partial dependence for.
        percentiles : tuple[float, float]
            Range of percentiles to evaluate (default 5th to 95th).
        n_points : int
            Number of evenly spaced points to evaluate.
        
        Returns
        -------
        tuple
            (x_values, y_values) for plotting.
        """
        if feature_name not in X_train.columns:
            raise ValueError(f"Feature '{feature_name}' not in dataset")
        
        feature_idx = list(X_train.columns).index(feature_name)
        
        pd_result = partial_dependence(
            self.model,
            X_train,
            [feature_idx],
            percentiles=percentiles,
            n_points=n_points,
        )
        
        x_values = pd_result["grid_values"][0]
        y_values = pd_result["average"][0]
        
        return x_values, y_values
    
    def summary_report(self, top_n: int = 10) -> str:
        """
        Generate a text summary of feature importance.
        
        Parameters
        ----------
        top_n : int
            Number of top features to include.
        
        Returns
        -------
        str
            Formatted importance report.
        """
        report = "FEATURE IMPORTANCE SUMMARY\n"
        report += "=" * 60 + "\n\n"
        
        if "tree_importance" in self.importance_scores:
            report += "Top Features (Tree-Based Importance):\n"
            report += "-" * 60 + "\n"
            df = self.importance_scores["tree_importance"].head(top_n)
            for _, row in df.iterrows():
                report += f"{row['Rank']:2d}. {row['Feature']:30s} - {row['Importance %']:6.2f}%\n"
            report += "\n"
        
        if "permutation_importance" in self.importance_scores:
            report += "Top Features (Permutation Importance):\n"
            report += "-" * 60 + "\n"
            df = self.importance_scores["permutation_importance"].head(top_n)
            for _, row in df.iterrows():
                report += f"{row['Rank']:2d}. {row['Feature']:30s} - {row['Importance %']:6.2f}%\n"
            report += "\n"
        
        if "shap_importance" in self.importance_scores:
            report += "Top Features (SHAP Importance):\n"
            report += "-" * 60 + "\n"
            df = self.importance_scores["shap_importance"].head(top_n)
            for _, row in df.iterrows():
                report += f"{row['Rank']:2d}. {row['Feature']:30s} - {row['SHAP %']:6.2f}%\n"
            report += "\n"
        
        return report
    
    def export_importance(self, filepath: str, importance_type: str = "tree"):
        """
        Export importance scores to CSV.
        
        Parameters
        ----------
        filepath : str
            Output CSV path.
        importance_type : str
            'tree', 'permutation', or 'shap'.
        """
        key_map = {
            "tree": "tree_importance",
            "permutation": "permutation_importance",
            "shap": "shap_importance",
        }
        
        key = key_map.get(importance_type)
        if key not in self.importance_scores:
            raise KeyError(f"No {importance_type} importance scores available")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.importance_scores[key].to_csv(filepath, index=False)
        print(f"[✓] Feature importance exported to {filepath}")


class ModelExplainer:
    """
    High-level model explanation and interpretation utilities.
    """
    
    @staticmethod
    def predict_with_explanation(
        model: Any,
        X: pd.DataFrame,
        sample_idx: int = 0,
        top_features: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate a local explanation for a single prediction.
        
        Parameters
        ----------
        model : sklearn estimator
        X : pd.DataFrame
            Dataset containing the sample.
        sample_idx : int
            Index of the sample to explain.
        top_features : int
            Number of top contributing features to return.
        
        Returns
        -------
        dict
            Prediction, probability, and feature contributions.
        """
        sample = X.iloc[sample_idx:sample_idx+1]
        prediction = model.predict(sample)[0]
        
        result = {
            "prediction": prediction,
            "sample": sample,
        }
        
        # Add probability if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(sample)[0]
            result["probability_class_0"] = proba[0]
            result["probability_class_1"] = proba[1]
        
        # Try to get feature contributions from tree models
        if hasattr(model, "predict") and hasattr(sample, "values"):
            result["input_values"] = sample.values[0]
            result["feature_names"] = sample.columns.tolist()
        
        return result
    
    @staticmethod
    def get_similar_predictions(
        model: Any,
        X: pd.DataFrame,
        sample_idx: int,
        n_similar: int = 5,
    ) -> pd.DataFrame:
        """
        Find similar predictions for a given sample.
        
        Parameters
        ----------
        model : sklearn estimator
        X : pd.DataFrame
            Training or full dataset.
        sample_idx : int
            Index of reference sample.
        n_similar : int
            Number of similar samples to find.
        
        Returns
        -------
        pd.DataFrame
            Samples with similar predictions.
        """
        reference_features = X.iloc[sample_idx].values
        
        # Calculate Euclidean distance to all other samples
        distances = np.sqrt(((X.values - reference_features) ** 2).sum(axis=1))
        
        # Get indices of n_similar + 1 closest (including self)
        similar_indices = np.argsort(distances)[:n_similar + 1][1:]  # Exclude self
        
        return X.iloc[similar_indices]
    
    @staticmethod
    def decision_path_summary(
        model: Any,
        explanation_type: str = "feature_importance",
    ) -> str:
        """
        Generate a summary of how the model makes decisions.
        
        Parameters
        ----------
        model : sklearn estimator
        explanation_type : str
            Type of explanation ('feature_importance', 'tree_depth', etc.).
        
        Returns
        -------
        str
            Text explanation of model decision-making.
        """
        summary = "Model Decision-Making Summary\n"
        summary += "=" * 50 + "\n\n"
        
        model_name = model.__class__.__name__
        summary += f"Model Type: {model_name}\n"
        
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            if hasattr(model, "n_estimators"):
                summary += f"Number of Trees: {model.n_estimators}\n"
            if hasattr(model, "max_depth"):
                summary += f"Max Depth: {model.max_depth}\n"
        
        summary += "\nHow This Model Works:\n"
        summary += "-" * 50 + "\n"
        
        if isinstance(model, RandomForestClassifier):
            summary += """
The Random Forest builds multiple decision trees using random subsets
of features. Each tree votes on the prediction, and the majority vote
wins. This ensemble approach reduces overfitting and improves robustness.
Most important features are those that provide the most information gain
when splitting at tree nodes.
            """
        elif isinstance(model, GradientBoostingClassifier):
            summary += """
Gradient Boosting sequentially builds decision trees, where each new tree
corrects the prediction errors of the previous ensemble. This sequential
correction focuses on improving difficult predictions, leading to high
accuracy. Important features are those that contribute most to reducing
the cumulative loss across all boosting iterations.
            """
        
        return summary.strip()


def evaluate_feature_stability(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Evaluate whether top important features are stable across train/test.
    
    Parameters
    ----------
    model : sklearn estimator (fitted)
    X_train, y_train, X_test, y_test : train/test split
    top_n : int
        Number of top features to check.
    
    Returns
    -------
    pd.DataFrame
        Stability report.
    """
    analyzer = FeatureImportanceAnalyzer(model)
    
    train_importance = analyzer.get_permutation_importance(X_train, y_train)
    test_importance = analyzer.get_permutation_importance(X_test, y_test)
    
    train_top = set(train_importance.head(top_n)["Feature"])
    test_top = set(test_importance.head(top_n)["Feature"])
    
    overlap = len(train_top & test_top)
    stability = overlap / top_n
    
    return pd.DataFrame({
        "Metric": ["Train Top Features", "Test Top Features", "Overlap", "Stability %"],
        "Value": [train_top, test_top, overlap, f"{stability*100:.1f}%"],
    })
