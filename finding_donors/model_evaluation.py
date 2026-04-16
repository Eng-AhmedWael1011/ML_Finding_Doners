"""
Model Evaluation Module
======================
Comprehensive evaluation metrics and reporting for classifier performance.

Provides:
  - Detailed confusion matrix analysis
  - Performance metrics (accuracy, precision, recall, F-score)
  - Cross-validation evaluation
  - Learning curves
  - Classification reports
"""

import numpy as np
import pandas as pd
from time import time
from typing import Dict, Tuple, Optional, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score, learning_curve


class ClassificationEvaluator:
    """
    Comprehensive evaluation toolkit for binary classifiers.
    
    Computes detailed metrics including accuracy, precision, recall, F-scores,
    confusion matrix, ROC-AUC, and generates classification reports.
    """
    
    def __init__(self, beta: float = 0.5):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        beta : float
            Beta parameter for F-beta score. Default 0.5 (emphasis on precision).
        """
        self.beta = beta
        self.results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        set_name: str = "test",
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation on predictions.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels (0 or 1).
        y_proba : array-like or None
            Predicted probabilities (used for ROC-AUC). If None, ROC-AUC skipped.
        set_name : str
            Dataset name (e.g., 'train', 'test', 'val').
        
        Returns
        -------
        dict
            Comprehensive metrics including accuracy, precision, recall,
            F-scores, confusion matrix, and ROC-AUC.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f_beta = fbeta_score(y_true, y_pred, beta=self.beta, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics = {
            "set": set_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            f"f{self.beta}_score": f_beta,
            "f1_score": f1,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "sensitivity": recall,
        }
        
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
                metrics["roc_auc"] = roc_auc
            except Exception as e:
                print(f"[WARNING] ROC-AUC calculation failed: {e}")
        
        self.results[set_name] = metrics
        return metrics
    
    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Print classification report (sklearn format).
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        """
        print(classification_report(y_true, y_pred, target_names=["<=50K", ">50K"]))
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get all evaluation results as a DataFrame for easy comparison.
        
        Returns
        -------
        pd.DataFrame
            Rows = datasets (train/test/val), Columns = metrics.
        """
        return pd.DataFrame(self.results).T
    
    def compare_classifiers(
        self,
        classifiers: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        cv: int = 5,
    ) -> pd.DataFrame:
        """
        Compare multiple classifiers using cross-validation and test set evaluation.
        
        Parameters
        ----------
        classifiers : dict
            Dictionary mapping classifier names to unfitted estimators.
        X_train, y_train : training data
        X_test, y_test : test data
        cv : int
            Number of cross-validation folds.
        
        Returns
        -------
        pd.DataFrame
            Comparison table with CV scores, test accuracy, and test F-score.
        """
        comparison = []
        
        for name, clf in classifiers.items():
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
            
            # Fit and test
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            acc_test = accuracy_score(y_test, y_pred)
            f_test = fbeta_score(y_test, y_pred, beta=self.beta)
            
            comparison.append({
                "Classifier": name,
                "CV Mean Accuracy": cv_scores.mean(),
                "CV Std": cv_scores.std(),
                "Test Accuracy": acc_test,
                f"Test F{self.beta}": f_test,
            })
        
        return pd.DataFrame(comparison)


class LearningCurveAnalyzer:
    """Generate and analyze learning curves for model diagnosis."""
    
    @staticmethod
    def plot_learning_curve(
        estimator,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        train_sizes: Optional[np.ndarray] = None,
        cv: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate learning curve data (handles sklearn's learning_curve).
        
        Parameters
        ----------
        estimator : sklearn estimator
        X_train, y_train : training data
        train_sizes : array[float] or None
            Training set sizes as fractions (0.0, 1.0].
            If None, defaults to [0.01, 0.1, 0.5, 0.8, 1.0].
        cv : int
            Cross-validation folds.
        
        Returns
        -------
        tuple
            (train_sizes_abs, train_scores, test_scores) where each score array
            has shape (len(train_sizes), cv).
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.01, 1.0, 10)
        
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator,
            X_train,
            y_train,
            train_sizes=train_sizes,
            cv=cv,
            n_jobs=-1,
        )
        
        return train_sizes_abs, train_scores, test_scores


def evaluate_with_timing(
    classifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    beta: float = 0.5,
) -> Dict[str, Any]:
    """
    Train a classifier and evaluate with timing information.
    
    Parameters
    ----------
    classifier : sklearn estimator
    X_train, y_train, X_test, y_test : train/test split
    beta : float
        F-beta beta parameter.
    
    Returns
    -------
    dict
        Metrics including train_time, predict_time, accuracy, F-score, etc.
    """
    evaluator = ClassificationEvaluator(beta=beta)
    
    # Train with timing
    start = time()
    classifier.fit(X_train, y_train)
    train_time = time() - start
    
    # Predict with timing
    start = time()
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    predict_time = time() - start
    
    # Get probabilities if available
    y_proba_test = None
    if hasattr(classifier, "predict_proba"):
        try:
            y_proba_test = classifier.predict_proba(X_test)[:, 1]
        except Exception:
            pass
    
    # Evaluate
    train_metrics = evaluator.evaluate(y_train, y_pred_train, set_name="train")
    test_metrics = evaluator.evaluate(y_test, y_pred_test, y_proba_test, set_name="test")
    
    train_metrics["train_time"] = train_time
    test_metrics["predict_time"] = predict_time
    
    return {
        "train": train_metrics,
        "test": test_metrics,
        "classifier": classifier,
    }
