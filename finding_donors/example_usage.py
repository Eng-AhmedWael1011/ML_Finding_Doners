"""
Example Usage Script
====================
Demonstrates how to use the various modules together for common tasks.

Run this script to see examples of:
  - Loading preprocessed data
  - Training and evaluating models
  - Optimizing hyperparameters
  - Analyzing feature importance
  - Generating visualizations
  - Saving and loading models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, fbeta_score

# Import our custom modules
from model_training import compute_naive_predictor, train_predict, CLASSIFIERS
from model_optimization import ModelOptimizer
from model_interpretation import FeatureImportanceAnalyzer
from model_evaluation import ClassificationEvaluator, evaluate_with_timing
from model_visualization import ModelVisualizer, create_evaluation_dashboard
from pipeline.model_export import ModelExporter, save_model_report


def example_1_load_and_preprocess():
    """Example 1: Load and preprocess data."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Load and Preprocess Data")
    print("="*70)
    
    # Load data
    data = pd.read_csv("census.csv")
    print(f"Loaded dataset: {data.shape[0]} samples × {data.shape[1]} features")
    
    # Print first few rows
    print("\nFirst 3 samples:")
    print(data.head(3))
    
    # Check preprocessing needs
    print("\nData info:")
    print(f"  Missing values: {data.isnull().sum().sum()}")
    print(f"  Duplicates: {data.duplicated().sum()}")
    print(f"  Categorical features: {data.select_dtypes(include='object').shape[1]}")
    print(f"  Numerical features: {data.select_dtypes(include='number').shape[1]}")
    
    # Separate target
    income_raw = data["income"]
    features_raw = data.drop("income", axis=1)
    
    # Log-transform skewed features
    skewed = ["capital-gain", "capital-loss"]
    features_log = pd.DataFrame(data=features_raw)
    features_log[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    features_scaled = pd.DataFrame(data=features_log)
    features_scaled[numerical] = scaler.fit_transform(features_log[numerical])
    
    # Encode features
    features_final = pd.get_dummies(features_scaled)
    income = income_raw.apply(lambda x: 1 if x == ">50K" else 0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_final, income, test_size=0.2, random_state=0
    )
    
    print(f"\nAfter preprocessing:")
    print(f"  Features: {features_final.shape[1]}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Class distribution (test): {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, features_final, income


def example_2_naive_baseline(y_test, income):
    """Example 2: Compute naive predictor baseline."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Naive Predictor Baseline")
    print("="*70)
    
    naive_results = compute_naive_predictor(income, beta=0.5)
    
    print(f"\nNaive Predictor (always predict '>50K'):")
    print(f"  True Positives: {naive_results['TP']}")
    print(f"  False Positives: {naive_results['FP']}")
    print(f"  True Negatives: {naive_results['TN']}")
    print(f"  False Negatives: {naive_results['FN']}")
    print(f"  Accuracy: {naive_results['accuracy']:.4f}")
    print(f"  Precision: {naive_results['precision']:.4f}")
    print(f"  Recall: {naive_results['recall']:.4f}")
    print(f"  F(0.5) Score: {naive_results['fscore']:.4f}")
    
    print("\nThis is our baseline - any real model must exceed these scores!")


def example_3_train_baseline_models(X_train, X_test, y_train, y_test):
    """Example 3: Train baseline models at different training set sizes."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Train Baseline Models")
    print("="*70)
    
    # Calculate sample sizes
    samples_100 = len(y_train)
    samples_10 = int(samples_100 * 0.10)
    samples_1 = int(samples_100 * 0.01)
    
    print(f"Training with sample sizes: {samples_1}, {samples_10}, {samples_100}")
    
    # Train Random Forest as example
    clf = CLASSIFIERS["clf_B"]  # Random Forest
    
    results = {}
    for i, n_samples in enumerate([samples_1, samples_10, samples_100]):
        print(f"\n→ Training on {n_samples} samples...")
        result = train_predict(clf, n_samples, X_train, y_train, X_test, y_test)
        results[i] = result
        print(f"  Train Time: {result['train_time']:.4f}s")
        print(f"  Predict Time: {result['pred_time']:.4f}s")
        print(f"  Train Accuracy: {result['acc_train']:.4f}")
        print(f"  Test Accuracy: {result['acc_test']:.4f}")
        print(f"  Train F-Score: {result['f_train']:.4f}")
        print(f"  Test F-Score: {result['f_test']:.4f}")


def example_4_evaluate_models(X_train, X_test, y_train, y_test):
    """Example 4: Compare and evaluate multiple models."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Model Evaluation & Comparison")
    print("="*70)
    
    evaluator = ClassificationEvaluator(beta=0.5)
    
    # Evaluate all three classifiers
    classifiers = {
        "Gradient Boosting": CLASSIFIERS["clf_A"],
        "Random Forest": CLASSIFIERS["clf_B"],
        "Logistic Regression": CLASSIFIERS["clf_C"]
    }
    
    print("\nEvaluating classifiers with cross-validation...")
    comparison = evaluator.compare_classifiers(
        classifiers, X_train, y_train, X_test, y_test, cv=5
    )
    
    print("\n" + comparison.to_string())
    
    # Get the best model
    best_idx = comparison["Test F0.5"].idxmax()
    best_model_name = comparison.loc[best_idx, "Classifier"]
    best_model = classifiers[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"  Test F-Score: {comparison.loc[best_idx, 'Test F0.5']:.4f}")
    
    return best_model, best_model_name


def example_5_optimize_model(best_model, best_model_name, X_train, y_train, X_test, y_test):
    """Example 5: Optimize hyperparameters of best model."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Hyperparameter Optimization")
    print("="*70)
    
    print(f"\nOptimizing {best_model_name}...")
    print("This may take a minute or two...\n")
    
    # Create optimizer with reduced grid for speed
    optimizer = ModelOptimizer(
        best_model_name,
        beta=0.5,
        cv=3,
        use_reduced_grid=True,  # Use smaller grid for example
    )
    
    # Run optimization
    optimized_clf, best_params, cv_score = optimizer.optimize_grid(
        X_train, y_train, verbose=0
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best CV F-Score: {cv_score:.4f}")
    print(f"  Best Parameters: {best_params}")
    
    # Evaluate on test set
    y_pred = optimized_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f = fbeta_score(y_test, y_pred, beta=0.5)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F-Score: {test_f:.4f}")
    
    return optimized_clf


def example_6_feature_importance(model, X_train, X_test, y_test):
    """Example 6: Analyze feature importance."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Feature Importance Analysis")
    print("="*70)
    
    analyzer = FeatureImportanceAnalyzer(model, X_train.columns.tolist())
    
    # Get different types of importance
    print("\nCalculating feature importance...")
    
    # Tree-based
    try:
        tree_imp = analyzer.get_tree_importance(X_train)
        print(f"✓ Tree-based importance: {len(tree_imp)} features")
    except:
        tree_imp = None
    
    # Permutation
    perm_imp = analyzer.get_permutation_importance(X_test, y_test, n_repeats=5)
    print(f"✓ Permutation importance: {len(perm_imp)} features")
    
    # Print top features
    print("\nTop 5 Most Important Features:")
    print(perm_imp[["Feature", "Importance %"]].head(5).to_string(index=False))
    
    return perm_imp


def example_7_visualize_results(model, X_train, X_test, y_train, y_test, importance_df):
    """Example 7: Create visualizations."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Generate Visualizations")
    print("="*70)
    
    from sklearn.metrics import confusion_matrix
    
    # Get predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Create a 2x2 visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Performance Dashboard", fontsize=16, fontweight="bold")
    
    # Confusion matrices
    ModelVisualizer.plot_confusion_matrix(
        y_train, y_pred_train, "Train Confusion Matrix", axes[0, 0]
    )
    ModelVisualizer.plot_confusion_matrix(
        y_test, y_pred_test, "Test Confusion Matrix", axes[0, 1]
    )
    
    # Feature importance
    ModelVisualizer.plot_feature_importance(
        importance_df, top_n=10, title="Feature Importance", ax=axes[1, 0]
    )
    
    # Metrics summary
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics_text = f"""
    TEST SET METRICS:
    ─────────────────
    Accuracy:   {accuracy_score(y_test, y_pred):.4f}
    Precision:  {precision_score(y_test, y_pred):.4f}
    Recall:     {recall_score(y_test, y_pred):.4f}
    F1-Score:   {f1_score(y_test, y_pred):.4f}
    
    INTERPRETATION:
    ───────────────
    • High precision: Avoid
      false alarms
    • High recall: Catch
      all positives
    • F1 balances both
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontfamily="monospace", fontsize=11)
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig("example_dashboard.png", dpi=150, bbox_inches="tight")
    print("\n✓ Visualization saved as 'example_dashboard.png'")
    plt.close()


def example_8_save_and_load_model(model):
    """Example 8: Save and load models."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Save and Load Models")
    print("="*70)
    
    exporter = ModelExporter("./example_models")
    
    # Save model
    model_path = exporter.save_model(
        model,
        "my_classifier_v1",
        metadata={
            "accuracy": 0.85,
            "f_score": 0.72,
            "description": "Example optimized model"
        }
    )
    print(f"✓ Model saved to: {model_path}")
    
    # Load model back
    loaded_model, metadata = exporter.load_model("my_classifier_v1")
    print(f"✓ Model loaded successfully")
    print(f"  Metadata: {metadata}")


def main():
    """Run all examples."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  FINDING DONORS - MODULE USAGE EXAMPLES".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        # Example 1: Data loading and preprocessing
        X_train, X_test, y_train, y_test, features_final, income = example_1_load_and_preprocess()
        
        # Example 2: Naive baseline
        example_2_naive_baseline(y_test, income)
        
        # Example 3: Train baseline models
        example_3_train_baseline_models(X_train, X_test, y_train, y_test)
        
        # Example 4: Model evaluation
        best_model, best_name = example_4_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Example 5: Optimize model
        optimized_model = example_5_optimize_model(best_model, best_name, X_train, y_train, X_test, y_test)
        
        # Example 6: Feature importance
        importance_df = example_6_feature_importance(optimized_model, X_train, X_test, y_test)
        
        # Example 7: Visualizations
        example_7_visualize_results(optimized_model, X_train, X_test, y_train, y_test, importance_df)
        
        # Example 8: Save/Load
        example_8_save_and_load_model(optimized_model)
        
        print("\n" + "="*70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nFor more advanced usage, see COMPREHENSIVE_README.md")
        print("To run the full pipeline, use: python full_pipeline.py")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
