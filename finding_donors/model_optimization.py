"""
Model Optimization Module
==========================
Hyperparameter tuning and model optimization for the Finding Donors project.

Provides:
  - GridSearchCV and RandomSearchCV wrappers
  - Hyperparameter optimization for all three base models
  - Model persistence (save/load optimized models)
  - Optimization history tracking
"""

import numpy as np
import pandas as pd
import pickle
from time import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Hyperparameter spaces for each classifier
PARAM_GRIDS = {
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.8, 0.9, 1.0],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [1000, 2000],
    },
}

# Reduced parameter spaces for faster testing
PARAM_GRIDS_REDUCED = {
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4],
    },
    "RandomForestClassifier": {
        "n_estimators": [100, 150],
        "max_depth": [15, 20],
        "min_samples_split": [2, 5],
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "max_iter": [1000, 2000],
    },
}


class ModelOptimizer:
    """
    Hyperparameter optimization wrapper for Finding Donors classifiers.
    
    Supports GridSearchCV and RandomizedSearchCV with F-beta scoring.
    """
    
    def __init__(
        self,
        classifier_name: str,
        beta: float = 0.5,
        cv: int = 5,
        use_reduced_grid: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        classifier_name : str
            One of: 'GradientBoostingClassifier', 'RandomForestClassifier', 'LogisticRegression'
        beta : float
            F-beta score beta parameter (default 0.5 emphasizes precision).
        cv : int
            Cross-validation folds.
        use_reduced_grid : bool
            If True, use a smaller parameter grid for faster optimization.
        random_state : int
            Random seed.
        n_jobs : int
            Number of parallel jobs (-1 = use all cores).
        """
        self.classifier_name = classifier_name
        self.beta = beta
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Select parameter grid
        if use_reduced_grid:
            self.param_grid = PARAM_GRIDS_REDUCED.get(classifier_name, {})
        else:
            self.param_grid = PARAM_GRIDS.get(classifier_name, {})
        
        self.grid_search = None
        self.best_estimator = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = {}
    
    def optimize_grid(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        verbose: int = 1,
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform GridSearchCV optimization.
        
        Parameters
        ----------
        X_train, y_train : training data
        verbose : int
            Verbosity level (0 = silent, 2 = detailed).
        
        Returns
        -------
        tuple
            (best_estimator, best_params, best_score)
        """
        print(f"\n{'='*70}")
        print(f"GridSearchCV Optimization: {self.classifier_name}")
        print(f"{'='*70}")
        print(f"Parameter grid size: {self._grid_size()}")
        print(f"CV folds: {self.cv}")
        
        # Create base classifier
        clf = self._create_base_classifier()
        
        # Create F-beta scorer
        scorer = make_scorer(fbeta_score, beta=self.beta)
        
        # GridSearchCV
        self.grid_search = GridSearchCV(
            clf,
            self.param_grid,
            scoring=scorer,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=verbose,
        )
        
        start = time()
        self.grid_search.fit(X_train, y_train)
        elapsed = time() - start
        
        self.best_estimator = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        self.best_score = self.grid_search.best_score_
        
        print(f"\n[COMPLETE] Optimization took {elapsed:.2f} seconds")
        print(f"Best F{self.beta} score (CV): {self.best_score:.4f}")
        print(f"\nBest parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        self.optimization_history["grid_search"] = {
            "method": "GridSearchCV",
            "elapsed_time": elapsed,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "cv_results": self.grid_search.cv_results_,
        }
        
        return self.best_estimator, self.best_params, self.best_score
    
    def optimize_random(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_iter: int = 20,
        verbose: int = 1,
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform RandomizedSearchCV optimization (faster for large spaces).
        
        Parameters
        ----------
        X_train, y_train : training data
        n_iter : int
            Number of parameter combinations to sample.
        verbose : int
            Verbosity level.
        
        Returns
        -------
        tuple
            (best_estimator, best_params, best_score)
        """
        print(f"\n{'='*70}")
        print(f"RandomizedSearchCV Optimization: {self.classifier_name}")
        print(f"{'='*70}")
        print(f"Iterations: {n_iter}")
        print(f"CV folds: {self.cv}")
        
        clf = self._create_base_classifier()
        scorer = make_scorer(fbeta_score, beta=self.beta)
        
        self.grid_search = RandomizedSearchCV(
            clf,
            self.param_grid,
            n_iter=n_iter,
            scoring=scorer,
            cv=self.cv,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=verbose,
        )
        
        start = time()
        self.grid_search.fit(X_train, y_train)
        elapsed = time() - start
        
        self.best_estimator = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        self.best_score = self.grid_search.best_score_
        
        print(f"\n[COMPLETE] Optimization took {elapsed:.2f} seconds")
        print(f"Best F{self.beta} score (CV): {self.best_score:.4f}")
        print(f"\nBest parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        self.optimization_history["random_search"] = {
            "method": "RandomizedSearchCV",
            "elapsed_time": elapsed,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "n_iter": n_iter,
        }
        
        return self.best_estimator, self.best_params, self.best_score
    
    def save_optimized_model(self, filepath: str):
        """
        Save the optimized model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        if self.best_estimator is None:
            raise ValueError("No optimized model found. Run optimize_grid() first.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "wb") as f:
            pickle.dump(self.best_estimator, f)
        
        print(f"[✓] Model saved to {filepath}")
    
    def load_optimized_model(self, filepath: str):
        """
        Load a previously optimized model.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
        """
        with open(filepath, "rb") as f:
            self.best_estimator = pickle.load(f)
        
        print(f"[✓] Model loaded from {filepath}")
        return self.best_estimator
    
    def get_best_estimator(self):
        """Return the optimized estimator."""
        return self.best_estimator
    
    def _create_base_classifier(self):
        """Create an unfit base classifier instance."""
        if self.classifier_name == "GradientBoostingClassifier":
            return GradientBoostingClassifier(random_state=self.random_state)
        elif self.classifier_name == "RandomForestClassifier":
            return RandomForestClassifier(random_state=self.random_state)
        elif self.classifier_name == "LogisticRegression":
            return LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_name}")
    
    def _grid_size(self) -> int:
        """Calculate total parameter combinations in grid."""
        size = 1
        for values in self.param_grid.values():
            size *= len(values)
        return size


def compare_optimized_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    classifiers: Optional[Dict[str, Any]] = None,
    beta: float = 0.5,
    cv: int = 5,
    use_reduced_grid: bool = False,
) -> pd.DataFrame:
    """
    Optimize multiple classifiers and compare results.
    
    Parameters
    ----------
    X_train, y_train, X_test, y_test : train/test split
    classifiers : dict or None
        Mapping of names to class objects. If None, uses defaults:
        GradientBoostingClassifier, RandomForestClassifier, LogisticRegression
    beta : float
        F-beta score beta.
    cv : int
        Cross-validation folds.
    use_reduced_grid : bool
        Use smaller parameter grids for faster optimization.
    
    Returns
    -------
    pd.DataFrame
        Comparison of optimized models on test set.
    """
    if classifiers is None:
        classifiers = {
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        }
    
    results = []
    
    for name in classifiers:
        print(f"\n{'—'*70}")
        optimizer = ModelOptimizer(
            name,
            beta=beta,
            cv=cv,
            use_reduced_grid=use_reduced_grid,
        )
        
        # Optimize
        best_clf, best_params, grid_score = optimizer.optimize_grid(X_train, y_train)
        
        # Evaluate on test set
        y_pred = best_clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_fscore = fbeta_score(y_test, y_pred, beta=beta)
        
        results.append({
            "Classifier": name,
            "CV F-Score": grid_score,
            "Test Accuracy": test_acc,
            f"Test F{beta}": test_fscore,
        })
    
    return pd.DataFrame(results)
