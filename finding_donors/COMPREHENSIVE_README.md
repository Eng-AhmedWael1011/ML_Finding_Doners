# Finding Donors for CharityML - Complete ML Project

A comprehensive machine learning project for predicting donor likelihood using census data. This project demonstrates a complete ML workflow from data preprocessing through model optimization and interpretation.

## 📋 Project Overview

**Goal**: Build a machine learning model to identify individuals making >$50K annually, helping CharityML target potential donors.

**Dataset**: Modified 1994 US Census dataset (~32,000 records, 13 features)

**Key Metrics**: F(0.5) score (emphasizes precision over recall)

## 📁 Project Structure

```
finding_donors/
├── finding_donors.ipynb              # Main notebook (interactive analysis)
├── full_pipeline.py                  # Complete ML pipeline runner
├── model-training.py                 # Baseline model training
├── model-optimization.py             # Hyperparameter tuning (GridSearchCV)
├── model_evaluation.py               # Evaluation metrics & utilities
├── model_interpretation.py           # Feature importance analysis
├── model_visualization.py            # Plotting utilities
├── requirements.txt                  # Python dependencies
├── census.csv                        # Raw data
├── census_preprocessed.csv           # Preprocessed data
├── visuals.py                        # Visualization helpers
│
├── pipeline/                         # Modular pipeline components
│   ├── __init__.py
│   ├── data_loader.py               # Load & inspect data
│   ├── eda.py                       # Exploratory data analysis
│   ├── preprocessing.py             # Data cleaning & transformation
│   ├── export.py                    # Export preprocessed data
│   ├── model_export.py              # Save/load models & pipelines
│   └── run_pipeline.py              # Execute full preprocessing
│
└── output/                           # Generated results (after running)
    ├── models/                      # Saved trained models
    ├── optimization_summary.csv     # Model comparison results
    ├── feature_importance_*.csv     # Feature rankings
    ├── *.png                        # Visualizations
    └── final_report.txt             # Comprehensive report
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Complete analysis (may take 5-10 minutes)
python full_pipeline.py

# Quick mode (reduced tuning, ~3-5 minutes)
python full_pipeline.py --quick

# Custom output directory
python full_pipeline.py --output ./my_results
```

### 3. Interactive Notebook

```bash
jupyter notebook finding_donors.ipynb
```

## 📊 Pipeline Stages

### Stage 1: Data Exploration & EDA

- Load and inspect Census dataset
- Check data types, missing values, duplicates
- Basic statistics and class distribution
- Identify skewed features

### Stage 2: Preprocessing

- Remove duplicates and handle missing values
- Log-transform skewed features (capital gain/loss)
- Min-Max scale numerical features (0-1 range)
- One-hot encode categorical features
- Train/test split (80/20)

### Stage 3: Baseline Model Training

Three classifiers evaluated at 1%, 10%, and 100% training data:

- **Gradient Boosting Classifier**: Sequential ensemble (typically best F-score)
- **Random Forest Classifier**: Parallel ensemble (fast, robust)
- **Logistic Regression**: Linear baseline (for comparison)

### Stage 4: Hyperparameter Optimization

GridSearchCV tuning for best-performing model:

- Gradient Boosting: n_estimators, learning_rate, max_depth, subsample
- Random Forest: n*estimators, max_depth, min_samples*\*
- Logistic Regression: C, penalty, solver

### Stage 5: Feature Importance Analysis

- Tree-based feature importance (.feature*importances*)
- Permutation importance (model-agnostic)
- SHAP values (if installed)
- Top 5-10 influential features identified

### Stage 6: Evaluation & Reporting

- Confusion matrix analysis
- Precision/Recall/F-score metrics
- ROC-AUC curves
- Learning curves
- Comprehensive report generation

## 📈 Key Metrics

| Metric        | Description                             | Target |
| ------------- | --------------------------------------- | ------ |
| **Accuracy**  | Correct predictions / Total predictions | ≥ 0.85 |
| **Precision** | True Positives / (TP + FP)              | ≥ 0.70 |
| **Recall**    | True Positives / (TP + FN)              | ≥ 0.55 |
| **F(0.5)**    | Weighted harmonic mean (precision×2)    | ≥ 0.65 |
| **ROC-AUC**   | Area under ROC curve                    | ≥ 0.80 |

## 🔍 Feature Analysis

Top predictive features typically include:

1. **Age** - Strong predictor of income
2. **Education-num** - Years of education completed
3. **Hours-per-week** - Full-time vs part-time indicator
4. **Capital-gain** - Investment income (highly skewed)
5. **Marital-status** - Family structure indicator

## 💻 Module Documentation

### model_training.py

Train baseline models and evaluate across different sample sizes.

**Key Functions:**

- `train_predict()` - Train models on sample sizes and time execution
- `run_evaluation()` - Train all classifiers at multiple training set sizes
- `main()` - Execute full training pipeline

### model-optimization.py

Hyperparameter tuning and model optimization.

**Key Classes:**

- `ModelOptimizer` - GridSearch/RandomSearch wrapper
- Methods: `optimize_grid()`, `optimize_random()`, `save_optimized_model()`

**Usage:**

```python
from model_optimization import ModelOptimizer

optimizer = ModelOptimizer("GradientBoostingClassifier", beta=0.5)
best_clf, best_params, score = optimizer.optimize_grid(X_train, y_train)
optimizer.save_optimized_model("./models/best_model.pkl")
```

### model_interpretation.py

Feature importance and model explanations.

**Key Classes:**

- `FeatureImportanceAnalyzer` - Multi-method feature importance
- Methods: `get_tree_importance()`, `get_permutation_importance()`, `get_shap_importance()`

**Usage:**

```python
from model_interpretation import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(model, X_train.columns)
importance_df = analyzer.get_permutation_importance(X_test, y_test)
print(analyzer.summary_report(top_n=10))
```

### model_evaluation.py

Comprehensive evaluation metrics.

**Key Classes:**

- `ClassificationEvaluator` - Confusion matrix, precision, recall, F-scores
- `LearningCurveAnalyzer` - Learning curve plotting

**Usage:**

```python
from model_evaluation import ClassificationEvaluator

evaluator = ClassificationEvaluator(beta=0.5)
metrics = evaluator.evaluate(y_test, y_pred)
print(evaluator.get_summary())
```

### model_visualization.py

Plotting and visualization utilities.

**Key Functions:**

- `plot_confusion_matrix()` - Confusion matrix heatmap
- `plot_feature_importance()` - Feature importance bars
- `plot_roc_curve()` - ROC curve
- `plot_model_comparison()` - Compare models
- `create_evaluation_dashboard()` - 2×2 dashboard

**Usage:**

```python
from model_visualization import ModelVisualizer, create_evaluation_dashboard

ModelVisualizer.plot_feature_importance(importance_df, top_n=15)
dashboard = create_evaluation_dashboard(y_train, y_pred_train, y_test, y_pred)
```

### pipeline/model_export.py

Save and load models with metadata.

**Key Classes:**

- `ModelExporter` - Save/load individual models
- `PipelineExporter` - Save complete preprocessing + model pipeline

**Usage:**

```python
from pipeline.model_export import ModelExporter, save_model_report

exporter = ModelExporter("./models")
exporter.save_model(model, "my_model", metadata={"accuracy": 0.85})

save_model_report(
    "GradientBoosting",
    metrics={"accuracy": 0.85, "f_score": 0.72},
    parameters={"n_estimators": 200},
    feature_importance=importance_df
)
```

## 📊 Output Files

After running `full_pipeline.py`, you'll get:

```
output/
├── optimization_summary.csv          # Model comparison table
├── feature_importance_tree.csv       # Tree-based importance
├── feature_importance_permutation.csv # Permutation importance
├── confusion_matrix.png              # Test set confusion matrix
├── feature_importance.png            # Top features bar plot
├── model_comparison.png              # Model performance comparison
├── evaluation_dashboard.png          # 4-panel evaluation dashboard
├── final_report.txt                  # Comprehensive text report
└── models/
    ├── GradientBoostingClassifier_optimized.pkl
    ├── GradientBoostingClassifier_optimized_metadata.json
    ├── RandomForestClassifier_optimized.pkl
    └── ... (other models)
```

## 🎯 Usage Examples

### Example 1: Quick Model Comparison

```python
from model_training import main, run_evaluation, CLASSIFIERS
from model_evaluation import ClassificationEvaluator
import pandas as pd

# Run training
accuracy, fscore, results = main(X_train, X_test, y_train, y_test, income)

# Evaluate
evaluator = ClassificationEvaluator(beta=0.5)
comparison = evaluator.compare_classifiers(
    CLASSIFIERS, X_train, y_train, X_test, y_test, cv=5
)
print(comparison)
```

### Example 2: Optimize Specific Model

```python
from model_optimization import ModelOptimizer

# Optimize Gradient Boosting
optimizer = ModelOptimizer("GradientBoostingClassifier", use_reduced_grid=False)
best_clf, best_params, score = optimizer.optimize_grid(X_train, y_train)

print(f"Best params: {best_params}")
print(f"CV F-Score: {score:.4f}")

# Save for later use
optimizer.save_optimized_model("./models/best_gb.pkl")
```

### Example 3: Analyze Feature Importance

```python
from model_interpretation import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(best_model, X_train.columns.tolist())

# Get multiple importance measures
tree_importance = analyzer.get_tree_importance(X_train)
perm_importance = analyzer.get_permutation_importance(X_test, y_test)

# Create visualizations
from model_visualization import ModelVisualizer
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ModelVisualizer.plot_feature_importance(tree_importance, ax=axes[0])
ModelVisualizer.plot_feature_importance(perm_importance, ax=axes[1])
plt.show()
```

### Example 4: Make Predictions on New Data

```python
from pipeline.model_export import ModelExporter

# Load a saved model
exporter = ModelExporter("./models")
model, metadata = exporter.load_model("GradientBoostingClassifier_optimized")

# Make predictions
new_data = pd.read_csv("new_donors.csv")
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)

# Save results
pd.DataFrame({
    "prediction": predictions,
    "probability_donor": probabilities[:, 1]
}).to_csv("donor_predictions.csv", index=False)
```

## 🔧 Customization

### Run with Different Models

```bash
python full_pipeline.py --models GradientBoostingClassifier,RandomForestClassifier
```

### Adjust Hyperparameter Space

Edit `model-optimization.py`:

```python
PARAM_GRIDS = {
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200, 300, 400],  # Add 400
        "learning_rate": [0.01, 0.02, 0.05, 0.1],  # Add 0.02
        # ... more parameters
    },
}
```

### Change Metrics Priority

```python
from model_evaluation import ClassificationEvaluator

# Emphasis on recall (catching all donors)
evaluator = ClassificationEvaluator(beta=1.0)  # F1-score

# More emphasis on precision (fewer false positives)
evaluator = ClassificationEvaluator(beta=0.2)
```

## 📚 References

- **Original Paper**: [Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf)
- **Dataset**: [UCI ML Repository - Census Income](https://archive.ics.uci.edu/ml/datasets/Census+Income)
- **Scikit-Learn**: [ML Library Documentation](https://scikit-learn.org)
- **F-Beta Score**: [Wikipedia - F-score](https://en.wikipedia.org/wiki/F-score)

## 🐛 Troubleshooting

### Error: "census.csv not found"

```bash
# Ensure you're in the finding_donors directory
cd finding_donors
python full_pipeline.py
```

### Memory Issues with Large Hyperparameter Grids

```bash
# Use quick mode with reduced grids
python full_pipeline.py --quick

# Or limit to specific model
python full_pipeline.py --models GradientBoostingClassifier
```

### ModuleNotFoundError

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📝 License & Citation

This project is part of the Udacity Machine Learning Engineer Nanodegree program.

**Data Citation**: Ron Kohavi and Barry Becker, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", _1996 KDD International Conference on Knowledge Discovery and Data Mining_.

## 👤 Author

Machine Learning Engineer Nanodegree Project

---

**Last Updated**: 2024  
**Python Version**: 3.8+  
**Status**: Complete & Production-Ready ✅
