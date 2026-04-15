# CharityML — Supervised Learning Project

import numpy as np
import pandas as pd
from time import time

from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Check for visuals.py
try:
    import visuals as vs

    HAS_VISUALS = True

except ImportError:
    HAS_VISUALS = False
    print("[WARNING] visuals.py not found — vs.evaluate() calls will be skipped.")


# NAIVE PREDICTOR BENCHMARK:
# --------------------------
#     Strategy: predict EVERYONE earns > 50K (always predict 1).
#     This is the worst-useful baseline — any real model must beat it.
#
#     With this strategy:
#       • TP = all actual positives  (we call every person a donor)
#       • FP = all actual negatives  (we wrongly call non-donors donors)
#       • TN = 0   (we never predict 0)
#       • FN = 0   (we never predict 0, so we never miss a positive)


def compute_naive_predictor(income: pd.Series, beta: float = 0.5) -> dict:
    """
    Compute the accuracy and F-beta score of a naive predictor that always
    predicts the positive class (income > 50K).

    Parameters
    ----------
    income : pd.Series
        Binary-encoded target label (0 = <=50K, 1 = >50K).
        This is the FULL dataset label — NOT just the test split.
    beta : float
        Beta value for the F-beta score formula. Default 0.5
        (weights precision twice as heavily as recall).

    Returns
    -------
    dict with keys: 'TP', 'FP', 'TN', 'FN', 'accuracy', 'recall',
                    'precision', 'fscore'
    """
    # Confusion Matrix values for the naive case
    TP = np.sum(income)  # every actual positive is "correctly" predicted
    FP = income.count() - TP  # every actual negative is wrongly called positive
    TN = 0  # we never predict 0
    FN = 0  # we never predict 0, so no missed positives

    # Metrics
    accuracy = TP / (TP + FP)  # = TP / total  (since TN=FN=0)
    recall = TP / (TP + FN)  # = 1.0  (we catch every positive)
    precision = TP / (TP + FP)  # same as accuracy in this naive case

    fscore = (
        (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    )  # F-beta formula

    results = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "fscore": fscore,
    }

    print(
        "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(
            accuracy, fscore
        )
    )

    return results


# TRAINING & PREDICTION PIPELINE:
# -------------------------------
#     - Trains a single learner on a SUBSET of training data, then evaluates it
#       on both a training slice (first 300 samples) and the full test set
#     - Timing is recorded for both the fit and predict phases


def train_predict(
    learner, sample_size: int, X_train, y_train, X_test, y_test, beta: float = 0.5
) -> dict:
    """
    Train a classifier on a slice of training data and evaluate it.

    Parameters
    ----------
    learner      : sklearn estimator — the model to train (unfitted).
    sample_size  : int — number of training rows to use (1 %, 10 %, or 100 %).
    X_train      : training features (full set; sliced internally).
    y_train      : training labels   (full set; sliced internally).
    X_test       : test features     (always evaluated on the full test set).
    y_test       : test labels       (always evaluated on the full test set).
    beta         : float — beta for F-beta score. Default 0.5.

    Returns
    -------
    dict with keys:
        train_time  — seconds to fit the model
        pred_time   — seconds to predict on test + first-300 train slice
        acc_train   — accuracy on first 300 training samples
        acc_test    — accuracy on full test set
        f_train     — F-beta score on first 300 training samples
        f_test      — F-beta score on full test set
    """
    results = {}

    # Fit on the sampled training slice
    start = time()
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    results["train_time"] = end - start

    # Predict on test set AND first 300 training samples
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()
    results["pred_time"] = end - start

    # Accuracy
    results["acc_train"] = accuracy_score(y_train[:300], predictions_train)
    results["acc_test"] = accuracy_score(y_test, predictions_test)

    # F-beta score
    results["f_train"] = fbeta_score(y_train[:300], predictions_train, beta=beta)
    results["f_test"] = fbeta_score(y_test, predictions_test, beta=beta)

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    return results


# CLASSIFIER DEFINITIONS:
# -----------------------
#     Three models are chosen deliberately to span the complexity spectrum:
#
#     - clf_A — GradientBoostingClassifier
#           Best real-world fit for tabular census data; sequential boosting
#           corrects residual errors tree-by-tree.  Slower to train but typically
#           highest F-score.  Good baseline for "best possible".
#
#     - clf_B — RandomForestClassifier
#           Parallel ensemble of trees — fast, robust to overfitting, naturally
#           handles our mix of one-hot categoricals + continuous numerics.
#           Also has .feature_importances_ (used in the feature importance stage).
#
#     - clf_C — LogisticRegression
#           Fast linear baseline.  If a complex model barely beats this, it signals
#           the decision boundary is simpler than expected.  Needs max_iter bump
#           because our one-hot feature space is wide (103 features after encoding).

RANDOM_STATE = 42

CLASSIFIERS = {
    "clf_A": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "clf_B": RandomForestClassifier(random_state=RANDOM_STATE),
    "clf_C": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
}


# FULL EVALUATION RUNNER:
# -----------------------
#     Trains each classifier at 1 %, 10 %, and 100 % of the training data
#     and collects results into a nested dict ready for vs.evaluate().


def run_evaluation(
    X_train, y_train, X_test, y_test, classifiers: dict = None, beta: float = 0.5
) -> dict:
    """
    Run train_predict for every classifier × every sample size.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : split dataset from preprocessing.
    classifiers : dict mapping name → sklearn estimator.
                  Defaults to the module-level CLASSIFIERS dict.
    beta        : F-beta beta value. Default 0.5.

    Returns
    -------
    results : nested dict  {clf_class_name: {0: {...}, 1: {...}, 2: {...}}}
              Keys 0, 1, 2 correspond to 1 %, 10 %, 100 % sample sizes.
              Passed directly to vs.evaluate().
    """
    if classifiers is None:
        classifiers = CLASSIFIERS

    # Sample-size thresholds
    samples_100 = len(y_train)
    samples_10 = int(samples_100 * 0.10)
    samples_1 = int(samples_100 * 0.01)
    sample_sizes = [samples_1, samples_10, samples_100]

    print(
        f"\nSample sizes  →  1%: {samples_1} | 10%: {samples_10} | 100%: {samples_100}\n"
    )

    results = {}
    for name, clf in classifiers.items():
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        print(f"── Evaluating {clf_name} ──────────────────────────")
        for i, n_samples in enumerate(sample_sizes):
            results[clf_name][i] = train_predict(
                clf, n_samples, X_train, y_train, X_test, y_test, beta=beta
            )
        print()

    return results


# MAIN:
# -----
#     - Called when you run:  python model_training.py
#     - Expects the preprocessing pipeline to have already produced:
#         X_train, X_test, y_train, y_test, income
#     - If run standalone, it re-runs preprocessing from census.csv so the file
#       is self-contained and testable on its own.


def _run_preprocessing(csv_path: str = "census.csv"):
    """
    Minimal inline preprocessing so this file can run standalone.
    In the notebook, preprocessing is already done by the pipeline module —
    this function is only used when executing model_training.py directly.
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(csv_path)

    income_raw = data["income"]
    features_raw = data.drop("income", axis=1)

    # Log-transform skewed features
    skewed = ["capital-gain", "capital-loss"]
    features_log = pd.DataFrame(data=features_raw)
    features_log[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

    # MinMax scale numerical features
    scaler = MinMaxScaler()
    numerical = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    features_log_minmax = pd.DataFrame(data=features_log)
    features_log_minmax[numerical] = scaler.fit_transform(features_log[numerical])

    # One-hot encode categoricals
    features_final = pd.get_dummies(features_log_minmax)

    # Encode target: >50K → 1, <=50K → 0
    income = income_raw.apply(lambda x: 1 if x == ">50K" else 0)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_final, income, test_size=0.2, random_state=0
    )

    return X_train, X_test, y_train, y_test, income


def main(
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    income=None,
    csv_path: str = "census.csv",
):
    """
    Full model-training stage.

    Can be called in two ways:

    1. From the notebook (data already preprocessed):
           from model_training import main
           accuracy, fscore, results = main(X_train, X_test, y_train, y_test, income)

    2. Standalone (self-contained preprocessing):
           python model_training.py
    """

    # If no data passed in, run preprocessing from scratch
    if any(v is None for v in [X_train, X_test, y_train, y_test, income]):
        print("[INFO] No data passed — running inline preprocessing from", csv_path)
        X_train, X_test, y_train, y_test, income = _run_preprocessing(csv_path)

    # Naive Predictor:
    # ----------------
    print("\n" + "=" * 60)
    print("SECTION 1 — NAIVE PREDICTOR BENCHMARK")
    print("=" * 60)
    naive = compute_naive_predictor(income, beta=0.5)

    # These two scalars are needed by vs.evaluate() and by the optimization stage
    accuracy = naive["accuracy"]
    fscore = naive["fscore"]

    # Train & evaluate three classifiers:
    # -----------------------------------
    print("\n" + "=" * 60)
    print("SECTION 2 — MODEL TRAINING & EVALUATION")
    print("=" * 60)
    results = run_evaluation(X_train, y_train, X_test, y_test)

    # Visualise (requires visuals.py in the same directory)
    if HAS_VISUALS:
        vs.evaluate(results, accuracy, fscore)
    else:
        print("\n[INFO] Skipping vs.evaluate() — visuals.py not available.")
        _print_results_table(results, accuracy, fscore)

    print("\n[DONE] model_training.py complete.")
    print("       → 'accuracy' and 'fscore' are the naive benchmark scalars.")
    print("       → 'results'  is the nested dict for vs.evaluate().")
    print("       → Pass all three to the optimization stage.\n")

    return accuracy, fscore, results


# Fallback text table (used when visuals.py is absent)
def _print_results_table(results: dict, naive_accuracy: float, naive_fscore: float):
    header = f"\n{'Model':<30} {'Sample':>8} {'Acc Train':>10} {'Acc Test':>10} {'F Train':>9} {'F Test':>9} {'Train(s)':>10} {'Pred(s)':>9}"
    print(header)
    print("-" * len(header))
    sample_labels = ["1%", "10%", "100%"]
    for clf_name, runs in results.items():
        for i, r in runs.items():
            print(
                f"{clf_name:<30} {sample_labels[i]:>8} "
                f"{r['acc_train']:>10.4f} {r['acc_test']:>10.4f} "
                f"{r['f_train']:>9.4f} {r['f_test']:>9.4f} "
                f"{r['train_time']:>10.4f} {r['pred_time']:>9.4f}"
            )
    print(
        f"\nNaive baseline  →  Accuracy: {naive_accuracy:.4f}  |  F-score: {naive_fscore:.4f}"
    )


if __name__ == "__main__":
    main()
