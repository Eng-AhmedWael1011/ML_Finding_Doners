"""
Model Export and Persistence Module
====================================
Save and load trained models with metadata for production use.

Provides:
  - Model serialization (pickle)
  - Metadata tracking
  - Model versioning
  - Pipeline export (including scaler, encoder)
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


class ModelExporter:
    """Handles model persistence and metadata tracking."""
    
    def __init__(self, output_dir: str = "./models"):
        """
        Initialize exporter.
        
        Parameters
        ----------
        output_dir : str
            Directory to save models.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Save a trained model with metadata.
        
        Parameters
        ----------
        model : sklearn estimator
            Trained model.
        name : str
            Model name (e.g., 'gradient_boosting_v1').
        metadata : dict or None
            Additional metadata (accuracy, parameters, etc.).
        overwrite : bool
            If False and file exists, raises error.
        
        Returns
        -------
        str
            Path to saved model.
        """
        model_path = self.output_dir / f"{name}.pkl"
        metadata_path = self.output_dir / f"{name}_metadata.json"
        
        if model_path.exists() and not overwrite:
            raise FileExistsError(f"Model already exists: {model_path}")
        
        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save metadata
        meta = metadata or {}
        meta["saved_at"] = datetime.now().isoformat()
        meta["model_class"] = model.__class__.__name__
        meta["model_file"] = str(model_path)
        
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        
        print(f"[✓] Model saved: {model_path}")
        print(f"[✓] Metadata saved: {metadata_path}")
        
        return str(model_path)
    
    def load_model(self, name: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a saved model with metadata.
        
        Parameters
        ----------
        name : str
            Model name.
        
        Returns
        -------
        tuple
            (model, metadata)
        """
        model_path = self.output_dir / f"{name}.pkl"
        metadata_path = self.output_dir / f"{name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        print(f"[✓] Model loaded: {model_path}")
        return model, metadata


class PipelineExporter:
    """Export complete preprocessing and modeling pipeline."""
    
    def __init__(self, output_dir: str = "./pipelines"):
        """Initialize pipeline exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_pipeline(
        self,
        scaler: Any,
        preprocessor: Any,
        model: Any,
        feature_names: List[str],
        target_name: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save complete pipeline (scaler + preprocessor + model).
        
        Parameters
        ----------
        scaler : sklearn.preprocessing.MinMaxScaler or similar
        preprocessor : sklearn.preprocessing.OneHotEncoder or similar
        model : sklearn estimator
        feature_names : list[str]
        target_name : str
        name : str
            Pipeline name.
        metadata : dict or None
        
        Returns
        -------
        str
            Path to saved pipeline.
        """
        pipeline_data = {
            "scaler": scaler,
            "preprocessor": preprocessor,
            "model": model,
            "feature_names": feature_names,
            "target_name": target_name,
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat(),
        }
        
        pipeline_path = self.output_dir / f"{name}_pipeline.pkl"
        
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline_data, f)
        
        print(f"[✓] Pipeline saved: {pipeline_path}")
        return str(pipeline_path)
    
    def load_pipeline(self, name: str) -> Dict[str, Any]:
        """
        Load a complete pipeline.
        
        Parameters
        ----------
        name : str
            Pipeline name.
        
        Returns
        -------
        dict
            Pipeline components: scaler, preprocessor, model, feature_names, target_name.
        """
        pipeline_path = self.output_dir / f"{name}_pipeline.pkl"
        
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        
        with open(pipeline_path, "rb") as f:
            pipeline_data = pickle.load(f)
        
        print(f"[✓] Pipeline loaded: {pipeline_path}")
        return pipeline_data


def save_predictions(
    predictions: pd.DataFrame,
    filepath: str,
    include_probabilities: bool = False,
):
    """
    Save predictions to CSV with optional probabilities.
    
    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions (with index matching original data).
    filepath : str
        Output CSV path.
    include_probabilities : bool
        If True, save prediction probabilities too.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(filepath, index=True)
    print(f"[✓] Predictions saved: {filepath}")


def save_model_report(
    model_name: str,
    metrics: Dict[str, float],
    parameters: Dict[str, Any],
    feature_importance: Optional[pd.DataFrame] = None,
    filepath: Optional[str] = None,
) -> str:
    """
    Generate and save a comprehensive model report.
    
    Parameters
    ----------
    model_name : str
        Name of the model.
    metrics : dict
        Performance metrics.
    parameters : dict
        Model hyperparameters.
    feature_importance : pd.DataFrame or None
        Feature importance table.
    filepath : str or None
        Output path. If None, uses default patterns.
    
    Returns
    -------
    str
        Path to saved report.
    """
    if filepath is None:
        filepath = f"./reports/{model_name}_report.txt"
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    report = f"""
{'='*70}
MODEL REPORT
{'='*70}

Model: {model_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE METRICS
{'-'*70}
"""
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            report += f"{metric:.<40} {value:.4f}\n"
        else:
            report += f"{metric:.<40} {value}\n"
    
    report += f"\nHYPERPARAMETERS\n{'-'*70}\n"
    for param, value in parameters.items():
        report += f"{param:.<40} {value}\n"
    
    if feature_importance is not None:
        report += f"\nTOP FEATURES\n{'-'*70}\n"
        for idx, row in feature_importance.head(10).iterrows():
            report += f"{idx+1}. {row['Feature']:.<35} {row.get('Importance %', row.get('Importance', 0)):>10}\n"
    
    report += f"\n{'='*70}\n"
    
    with open(filepath, "w") as f:
        f.write(report)
    
    print(f"[✓] Report saved: {filepath}")
    return filepath
