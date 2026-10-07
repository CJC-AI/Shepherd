"""Evaluate the frozen Shepherd fraud threshold on the held-out test set.

This module loads the trained XGBoost baseline model, evaluates it on the
chronologically held-out test split, and reports classification metrics using
the operating threshold selected previously on the validation set.

The threshold is intentionally fixed at 0.85. It must not be optimized using
the test set because the test split is reserved for final model evaluation.

This script is intended to answer:

    "How does the finalized baseline fraud detector perform on later,
     previously unseen transactions?"

Usage:
    python -m training.evaluate_thresholds
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


MODEL_PATH = Path("artifacts/models/xgboost_baseline.json")
FEATURE_DATASET_PATH = Path(
    "artifacts/features/transaction_features.parquet"
)

TARGET_COLUMN = "is_fraud"

# This threshold was selected using validation data.
# It is now frozen for final test evaluation.
OPERATING_THRESHOLD = 0.85

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

MODEL_FEATURES = [
    "amount",
    "merchant_risk_score",
    "is_home_country",
    "device_age_hours",
    "transactions_before",
    "avg_amount_before",
    "amount_vs_avg",
    "seconds_since_previous_tx",
    "tx_count_10m",
    "tx_count_1h",
    "tx_count_24h",
    "merchant_category_seen_before",
]


def load_test_split() -> tuple[pd.DataFrame, pd.Series]:
    """Load the chronological held-out test split.

    Returns:
        A tuple containing:
            X_test: Model feature matrix for the test period.
            y_test: Fraud labels for the test period.

    Raises:
        FileNotFoundError:
            If the feature dataset does not exist.
        ValueError:
            If the dataset is empty or the expected target column is missing.
    """
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATASET_PATH}"
        )

    df = pd.read_parquet(FEATURE_DATASET_PATH)

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from the dataset."
        )

    total_rows = len(df)

    train_end = int(total_rows * TRAIN_FRACTION)
    validation_end = train_end + int(
        total_rows * VALIDATION_FRACTION
    )

    test_df = df.iloc[validation_end:].copy()

    if test_df.empty:
        raise ValueError("Chronological test split is empty.")

    X_test = test_df[MODEL_FEATURES].copy()
    y_test = test_df[TARGET_COLUMN].astype(int)

    # The training pipeline replaces this expected first-transaction null
    # with zero because there is no previous transaction to measure against.
    X_test["seconds_since_previous_tx"] = (
        X_test["seconds_since_previous_tx"].fillna(0.0)
    )

    X_test = X_test.astype(np.float32)

    return X_test, y_test


def evaluate_test_set(
    model: xgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | int]:
    """Evaluate the frozen operating threshold on the test set.

    Args:
        model: Trained XGBoost booster.
        X_test: Test feature matrix.
        y_test: Ground-truth fraud labels.

    Returns:
        Dictionary containing ranking and threshold-dependent metrics.

    Raises:
        ValueError:
            If the number of predictions does not match the number of labels.
    """
    test_matrix = xgb.DMatrix(X_test)

    probabilities = model.predict(test_matrix)

    if len(probabilities) != len(y_test):
        raise ValueError(
            "Prediction count does not match test label count."
        )

    predictions = (
        probabilities >= OPERATING_THRESHOLD
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    total_rows = len(y_test)
    fraud_count = int(y_test.sum())
    non_fraud_count = total_rows - fraud_count
    alerts = int(predictions.sum())

    false_positive_rate = (
        fp / non_fraud_count
        if non_fraud_count > 0
        else 0.0
    )

    fraud_capture_rate = (
        tp / fraud_count
        if fraud_count > 0
        else 0.0
    )

    alert_rate = (
        alerts / total_rows
        if total_rows > 0
        else 0.0
    )

    return {
        "rows": total_rows,
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "threshold": OPERATING_THRESHOLD,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "false_positive_rate": false_positive_rate,
        "fraud_capture_rate": fraud_capture_rate,
        "alert_rate": alert_rate,
        "alerts": alerts,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def print_results(
    metrics: dict[str, float | int],
) -> None:
    """Print the final held-out test evaluation in a readable format.

    Args:
        metrics: Dictionary returned by `evaluate_test_set`.
    """
    print("Final model evaluation: TEST")
    print()
    print(f"Rows evaluated: {metrics['rows']:,}")
    print(
        f"Fraud prevalence: "
        f"{metrics['fraud_count'] / metrics['rows']:.4%}"
    )
    print(f"Operating threshold: {metrics['threshold']:.2f}")
    print()

    print("Ranking metrics")
    print("----------------")
    print(f"PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print()

    print("Threshold metrics")
    print("------------------")
    print(f"Precision:            {metrics['precision']:.4%}")
    print(f"Recall:               {metrics['recall']:.4%}")
    print(f"F1:                   {metrics['f1']:.4f}")
    print(
        f"False-positive rate:  "
        f"{metrics['false_positive_rate']:.4%}"
    )
    print(
        f"Fraud capture rate:   "
        f"{metrics['fraud_capture_rate']:.4%}"
    )
    print(
        f"Alert rate:           "
        f"{metrics['alert_rate']:.4%}"
    )
    print(f"Alerts generated:     {metrics['alerts']:,}")
    print()

    print("Confusion matrix")
    print("-----------------")
    print(f"True positives:       {metrics['true_positives']:,}")
    print(f"False positives:      {metrics['false_positives']:,}")
    print(f"False negatives:      {metrics['false_negatives']:,}")
    print(f"True negatives:       {metrics['true_negatives']:,}")


def main() -> None:
    """Run the final held-out test evaluation."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    print("Loading test data...")
    X_test, y_test = load_test_split()

    print(f"Loading model: {MODEL_PATH}")
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))

    print("Evaluating frozen threshold...")
    metrics = evaluate_test_set(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    print()
    print_results(metrics)


if __name__ == "__main__":
    main()