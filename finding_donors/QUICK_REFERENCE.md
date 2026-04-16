# Quick Reference Guide

## 🚀 Getting Started (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick pipeline (2-3 minutes)
python full_pipeline.py --quick

# View results
ls -la output/
```

## 📚 Module Import Cheat Sheet

### Model Training

```python
from model_training import train_predict, run_evaluation, CLASSIFIERS

# Train a model on a sample
results = train_predict(clf, sample_size, X_train, y_train, X_test, y_test)

# Evaluate all classifiers
results = run_evaluation(X_train, y_train, X_test, y_test)
```

### Optimization

```python
from model_optimization import ModelOptimizer

optimizer = ModelOptimizer("GradientBoostingClassifier")
best_clf, best_params, score = optimizer.optimize_grid(X_train, y_train)
optimizer.save_optimized_model("./models/best.pkl")
```

### Interpretation

```python
from model_interpretation import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(model, X_train.columns)
tree_imp = analyzer.get_tree_importance(X_train)
perm_imp = analyzer.get_permutation_importance(X_test, y_test)
print(analyzer.summary_report(top_n=10))
```

### Evaluation

```python
from model_evaluation import ClassificationEvaluator

evaluator = ClassificationEvaluator(beta=0.5)
metrics = evaluator.evaluate(y_test, y_pred, y_proba)
comparison = evaluator.compare_classifiers(classifiers_dict, X_train, y_train, X_test, y_test)
```

### Visualization

```python
from model_visualization import ModelVisualizer
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ModelVisualizer.plot_confusion_matrix(y_test, y_pred, ax=ax)
ModelVisualizer.plot_feature_importance(importance_df, top_n=15)
plt.show()
```

### Model Persistence

```python
from pipeline.model_export import ModelExporter, save_model_report

exporter = ModelExporter("./models")
exporter.save_model(model, "my_model", metadata={"score": 0.85})
model, meta = exporter.load_model("my_model")

save_model_report("GradientBoosting", metrics, params, importance_df)
```

## 🎯 Common Tasks

### 1. Quick Model Comparison

```python
from model_evaluation import ClassificationEvaluator
from model_training import CLASSIFIERS

evaluator = ClassificationEvaluator()
comp = evaluator.compare_classifiers(
    CLASSIFIERS, X_train, y_train, X_test, y_test
)
print(comp)
```

### 2. Optimize a Specific Model

```python
from model_optimization import ModelOptimizer

opt = ModelOptimizer("RandomForestClassifier")
clf, params, score = opt.optimize_grid(X_train, y_train)
print(f"Best F-Score: {score:.4f}")
```

### 3. Get Top 5 Features

```python
from model_interpretation import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(model, X_train.columns)
imp = analyzer.get_permutation_importance(X_test, y_test)
top_5 = analyzer.get_top_features("permutation", top_n=5)
print(top_5)
```

### 4. Train Model with Timing

```python
from model_evaluation import evaluate_with_timing

results = evaluate_with_timing(model, X_train, y_train, X_test, y_test)
print(f"Train time: {results['train']['train_time']:.2f}s")
print(f"Test accuracy: {results['test']['accuracy']:.4f}")
```

### 5. Create Evaluation Dashboard

```python
from model_visualization import create_evaluation_dashboard

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

fig = create_evaluation_dashboard(
    y_train, y_pred_train, y_test, y_pred, y_proba
)
fig.savefig("dashboard.png")
```

### 6. Analyze Feature Stability

```python
from model_interpretation import evaluate_feature_stability

stability = evaluate_feature_stability(
    model, X_train, y_train, X_test, y_test, top_n=5
)
print(stability)
```

## ⚙️ Configuration

### Hyperparameter Grids

Edit `model-optimization.py`:

```python
PARAM_GRIDS = {
    "GradientBoostingClassifier": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
    },
}
```

### Scoring Preference

```python
# Emphasize precision (fewer false positives)
evaluator = ClassificationEvaluator(beta=0.2)

# Emphasis on recall (catch more positives)
evaluator = ClassificationEvaluator(beta=2.0)

# Balanced (F1-score)
evaluator = ClassificationEvaluator(beta=1.0)
```

### Output Directory

```bash
python full_pipeline.py --output ./my_results
```

## 📊 Output Files Explained

| File                                 | Contains                    |
| ------------------------------------ | --------------------------- |
| `optimization_summary.csv`           | Model comparison results    |
| `feature_importance_tree.csv`        | Tree-based feature rankings |
| `feature_importance_permutation.csv` | Permutation importance      |
| `confusion_matrix.png`               | Predicted vs actual labels  |
| `feature_importance.png`             | Top features bar plot       |
| `model_comparison.png`               | Performance comparison      |
| `evaluation_dashboard.png`           | 4-panel results dashboard   |
| `final_report.txt`                   | Comprehensive text report   |
| `models/*.pkl`                       | Saved trained models        |

## 🐛 Quick Fixes

### Out of Memory

```bash
# Use quick mode
python full_pipeline.py --quick

# Or optimize specific model
python full_pipeline.py --models GradientBoostingClassifier
```

### Slow GridSearch

```python
# Use reduced grid
optimizer = ModelOptimizer("GradientBoost", use_reduced_grid=True)

# Or RandomSearch instead
optimizer.optimize_random(X_train, y_train, n_iter=10)
```

### Import Errors

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check Python version (3.8+)
python --version
```

## 🎓 Learning Path

1. **Start**: Run `python full_pipeline.py --quick`
2. **Explore**: Review generated outputs in `output/`
3. **Learn**: Read `COMPREHENSIVE_README.md`
4. **Experiment**: Run `example_usage.py`
5. **Customize**: Edit `PARAM_GRIDS` and re-run
6. **Deploy**: Use `pipeline/model_export.py`

## 📞 Key Functions (Alphabetical)

```
ClassificationEvaluator.compare_classifiers()
ClassificationEvaluator.evaluate()
ClassificationEvaluator.get_summary()
ClassificationEvaluator.print_report()

create_evaluation_dashboard()

FeatureImportanceAnalyzer.export_importance()
FeatureImportanceAnalyzer.get_partial_dependence()
FeatureImportanceAnalyzer.get_permutation_importance()
FeatureImportanceAnalyzer.get_shap_importance()
FeatureImportanceAnalyzer.get_top_features()
FeatureImportanceAnalyzer.get_tree_importance()
FeatureImportanceAnalyzer.summary_report()

LearningCurveAnalyzer.plot_learning_curve()

ModelExplainer.decision_path_summary()
ModelExplainer.get_similar_predictions()
ModelExplainer.predict_with_explanation()

ModelExporter.load_model()
ModelExporter.save_model()

ModelOptimizer.get_best_estimator()
ModelOptimizer.load_optimized_model()
ModelOptimizer.optimize_grid()
ModelOptimizer.optimize_random()
ModelOptimizer.save_optimized_model()

ModelVisualizer.plot_confusion_matrix()
ModelVisualizer.plot_feature_importance()
ModelVisualizer.plot_learning_curve()
ModelVisualizer.plot_metrics_comparison()
ModelVisualizer.plot_model_comparison()
ModelVisualizer.plot_partial_dependence()
ModelVisualizer.plot_roc_curve()

run_evaluation()

save_model_report()

train_predict()

PipelineExporter.load_pipeline()
PipelineExporter.save_pipeline()
```

## 💡 Pro Tips

✨ **Tip 1**: Use `beta=0.5` to emphasize precision (reduce false alarms)

✨ **Tip 2**: Start with `--quick` mode, then run full mode

✨ **Tip 3**: Save models to avoid retraining: `model_exporter.save_model(clf, "name")`

✨ **Tip 4**: Compare multiple importance methods for robust feature analysis

✨ **Tip 5**: Use permutation importance for model-agnostic results

✨ **Tip 6**: Check learning curves to diagnose overfitting

✨ **Tip 7**: Always evaluate on test set (not training set)

✨ **Tip 8**: Save optimization history for reproducibility

---

**Last Updated**: 2024  
**Python**: 3.8+  
**Status**: Ready to Use ✅
