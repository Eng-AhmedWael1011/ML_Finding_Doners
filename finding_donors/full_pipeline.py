"""
Full ML Pipeline Runner
=======================
Complete end-to-end machine learning pipeline for Finding Donors project.

Stages:
  1. Data Exploration & EDA
  2. Data Preprocessing
  3. Model Training (baseline comparison)
  4. Model Optimization (hyperparameter tuning)
  5. Feature Importance Analysis
  6. Final Evaluation & Reporting

Usage:
    python full_pipeline.py [--quick] [--models MODEL1,MODEL2] [--output DIR]
"""

import argparse
import sys
from pathlib import Path

# Add finding_donors directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model_training import main as train_models
from model_optimization import ModelOptimizer, PARAM_GRIDS_REDUCED
from model_interpretation import FeatureImportanceAnalyzer
from model_evaluation import ClassificationEvaluator, evaluate_with_timing
from model_visualization import ModelVisualizer, create_evaluation_dashboard
from pipeline.model_export import ModelExporter, save_model_report
from pipeline.run_pipeline import run as run_preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, fbeta_score


def run_full_pipeline(
    csv_path: str = "census.csv",
    quick_mode: bool = False,
    selected_models: list = None,
    output_dir: str = None,
):
    """
    Execute complete ML pipeline.
    
    Parameters
    ----------
    csv_path : str
        Path to census CSV file.
    quick_mode : bool
        If True, uses reduced hyperparameter grids for speed.
    selected_models : list or None
        Model names to optimize. If None, uses all three.
    output_dir : str or None
        Directory for outputs. Default: './output'.
    """
    
    if output_dir is None:
        output_dir = "./output"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(" "*20 + "FINDING DONORS - FULL ML PIPELINE")
    print("="*80 + "\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: Preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("STAGE 1: DATA EXPLORATION & PREPROCESSING")
    print("─"*80 + "\n")
    
    try:
        preprocessing_results = run_preprocessing(
            csv_path=csv_path,
            show_plots=False,
            export_csv=True,
            output_path=str(output_path / "census_preprocessed.csv")
        )
        
        X_train = preprocessing_results[0]  # features_final (will be split)
        income = preprocessing_results[1]
        
        # Re-split from the run_pipeline
        from sklearn.model_selection import train_test_split
        features_final = X_train
        X_train, X_test, y_train, y_test = train_test_split(
            features_final, income, test_size=0.2, random_state=0
        )
        
        print(f"\n[✓] Data preprocessing complete")
        print(f"    Training samples: {X_train.shape[0]}")
        print(f"    Test samples: {X_test.shape[0]}")
        print(f"    Features: {X_train.shape[1]}")
        
    except Exception as e:
        print(f"[!] Preprocessing failed: {e}")
        print("    Attempting fallback preprocessing...")
        
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        
        data = pd.read_csv(csv_path)
        income_raw = data["income"]
        features_raw = data.drop("income", axis=1)
        
        # Log transform
        skewed = ["capital-gain", "capital-loss"]
        features_log = pd.DataFrame(data=features_raw)
        features_log[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
        
        # Scale
        scaler = MinMaxScaler()
        numerical = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        features_scaled = pd.DataFrame(data=features_log)
        features_scaled[numerical] = scaler.fit_transform(features_log[numerical])
        
        # Encode
        features_final = pd.get_dummies(features_scaled)
        income = income_raw.apply(lambda x: 1 if x == ">50K" else 0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_final, income, test_size=0.2, random_state=0
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Initial Model Training
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("STAGE 2: BASELINE MODEL TRAINING")
    print("─"*80 + "\n")
    
    try:
        # Run baseline model training
        results = train_models(X_train, X_test, y_train, y_test, income)
        print(f"[✓] Baseline training complete")
    except Exception as e:
        print(f"[!] Baseline training error: {e}")
        results = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: Model Optimization
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("STAGE 3: HYPERPARAMETER OPTIMIZATION")
    print("─"*80 + "\n")
    
    if selected_models is None:
        selected_models = [
            "GradientBoostingClassifier",
            "RandomForestClassifier",
            "LogisticRegression"
        ]
    
    model_exporter = ModelExporter(output_path / "models")
    optimized_models = {}
    optimization_summary = []
    
    for model_name in selected_models:
        print(f"\n>>> Optimizing {model_name}...")
        
        optimizer = ModelOptimizer(
            model_name,
            beta=0.5,
            cv=5,
            use_reduced_grid=quick_mode,
        )
        
        try:
            best_clf, best_params, grid_score = optimizer.optimize_grid(X_train, y_train, verbose=0 if quick_mode else 1)
            
            # Evaluate on test set
            y_pred = best_clf.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_f05 = fbeta_score(y_test, y_pred, beta=0.5)
            
            optimized_models[model_name] = best_clf
            optimization_summary.append({
                "Model": model_name,
                "CV F-Score": grid_score,
                "Test Accuracy": test_acc,
                "Test F(0.5)": test_f05,
            })
            
            # Save model
            model_exporter.save_model(
                best_clf,
                f"{model_name}_optimized",
                metadata={
                    "best_params": best_params,
                    "cv_score": float(grid_score),
                    "test_accuracy": float(test_acc),
                    "test_f_score": float(test_f05),
                }
            )
            
            print(f"    ✓ Best CV F-Score: {grid_score:.4f}")
            print(f"    ✓ Test Accuracy: {test_acc:.4f}")
            print(f"    ✓ Test F-Score: {test_f05:.4f}")
            
        except Exception as e:
            print(f"    [!] Optimization failed: {e}")
    
    opt_summary_df = pd.DataFrame(optimization_summary)
    opt_summary_df.to_csv(output_path / "optimization_summary.csv", index=False)
    print(f"\n[✓] Optimization summary saved")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4: Feature Importance Analysis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("STAGE 4: FEATURE IMPORTANCE ANALYSIS")
    print("─"*80 + "\n")
    
    # Use the best model (highest F-score)
    best_model_row = opt_summary_df.loc[opt_summary_df["Test F(0.5)"].idxmax()]
    best_model_name = best_model_row["Model"]
    best_model = optimized_models[best_model_name]
    
    print(f"Analyzing features with: {best_model_name}\n")
    
    analyzer = FeatureImportanceAnalyzer(best_model, X_train.columns.tolist())
    
    # Tree importance
    tree_importance = analyzer.get_tree_importance(X_train)
    if not tree_importance.empty:
        tree_importance.to_csv(output_path / "feature_importance_tree.csv", index=False)
        print("✓ Tree-based importance calculated")
    
    # Permutation importance
    perm_importance = analyzer.get_permutation_importance(X_test, y_test, n_repeats=10)
    perm_importance.to_csv(output_path / "feature_importance_permutation.csv", index=False)
    print("✓ Permutation importance calculated")
    
    # Print summary
    print("\n" + analyzer.summary_report(top_n=10))
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5: Detailed Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("STAGE 5: DETAILED EVALUATION")
    print("─"*80 + "\n")
    
    evaluator = ClassificationEvaluator(beta=0.5)
    
    y_pred = best_model.predict(X_test)
    y_proba = None
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_metrics = evaluator.evaluate(y_test, y_pred, y_proba, set_name="test")
    
    print("Test Set Metrics:")
    print(f"  Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"  Precision:  {test_metrics['precision']:.4f}")
    print(f"  Recall:     {test_metrics['recall']:.4f}")
    print(f"  F(0.5):     {test_metrics['f0.5_score']:.4f}")
    print(f"  F1-Score:   {test_metrics['f1_score']:.4f}")
    if 'roc_auc' in test_metrics:
        print(f"  ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 6: Visualizations
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("STAGE 6: GENERATING VISUALIZATIONS")
    print("─"*80 + "\n")
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ModelVisualizer.plot_confusion_matrix(y_test, y_pred, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("✓ Confusion matrix plot saved")
    
    # Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    ModelVisualizer.plot_feature_importance(perm_importance, top_n=15, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path / "feature_importance.png", dpi=150, bbox_inches="tight")
    print("✓ Feature importance plot saved")
    
    # Model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ModelVisualizer.plot_model_comparison(opt_summary_df, metric="Test F(0.5)", ax=ax)
    fig.tight_layout()
    fig.savefig(output_path / "model_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Model comparison plot saved")
    
    # Dashboard
    y_pred_train = best_model.predict(X_train)
    dashboard = create_evaluation_dashboard(y_train, y_pred_train, y_test, y_pred, y_proba)
    dashboard.savefig(output_path / "evaluation_dashboard.png", dpi=150, bbox_inches="tight")
    print("✓ Evaluation dashboard saved")
    
    plt.close("all")
    
    # ─────────────────────────────────────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("GENERATING FINAL REPORT")
    print("─"*80 + "\n")
    
    report_path = save_model_report(
        best_model_name,
        test_metrics,
        best_model.get_params() if hasattr(best_model, "get_params") else {},
        feature_importance=perm_importance,
        filepath=str(output_path / "final_report.txt")
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(" "*30 + "PIPELINE COMPLETE")
    print("="*80)
    print(f"\n📊 Results saved to: {output_path}")
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"✓ Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"✓ Test F(0.5): {test_metrics['f0.5_score']:.4f}")
    print(f"\nOutput files:")
    print(f"  - optimization_summary.csv")
    print(f"  - feature_importance_tree.csv")
    print(f"  - feature_importance_permutation.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - feature_importance.png")
    print(f"  - model_comparison.png")
    print(f"  - evaluation_dashboard.png")
    print(f"  - final_report.txt")
    print(f"  - models/")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finding Donors - Full ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python full_pipeline.py                 # Run full pipeline
  python full_pipeline.py --quick         # Run quick mode (reduced tuning)
  python full_pipeline.py --output ./results
        """
    )
    
    parser.add_argument(
        "--csv", "-c",
        default="census.csv",
        help="Path to census CSV file (default: census.csv)"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Use reduced hyperparameter grids for faster execution"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    parser.add_argument(
        "--models", "-m",
        default=None,
        help="Comma-separated model names to optimize (default: all three)"
    )
    
    args = parser.parse_args()
    
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    
    run_full_pipeline(
        csv_path=args.csv,
        quick_mode=args.quick,
        selected_models=models,
        output_dir=args.output,
    )
