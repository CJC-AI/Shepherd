
"""Evaluate Shepherd's sigmoid-calibrated model on the reserved test period.

This module loads the model and sigmoid calibrator from the calibration
experiment and evaluates them on the final 15% of the chronologically
ordered feature dataset.

The operating threshold is fixed at 0.23. It was selected using validation
data under the policy of maximizing recall while meeting a minimum precision
target of 50%.

The threshold is not optimized or changed using test labels.

Reported metrics include:
    - PR-AUC and ROC-AUC for ranking performance.
    - Brier score, log loss, and expected calibration error for probability
      quality.
    - Precision, recall, F1, false-positive rate, fraud capture rate,
      alert rate, and confusion-matrix counts at the frozen threshold.

The script saves its results to a separate JSON artifact. It does not modify
the model, calibrator, feature dataset, or operating threshold.

Important:
    The final period was reserved by this calibration experiment, but it was
    previously examined during the original baseline experiment. Therefore,
    it is not a completely untouched test period across the entire project.

Usage:
    python -m training.evaluate_calibrated_test
"""

import gc
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURE_DATASET_PATH = Path(
    "artifacts/features/transaction_features.parquet"
)

MODEL_PATH = Path(
    "artifacts/models/xgboost_calibration_base.json"
)

CALIBRATORS_PATH = Path(
    "artifacts/models/calibrators.joblib"
)

FEATURE_LIST_PATH = Path(
    "artifacts/models/xgboost_calibration_base_features.json"
)

OUTPUT_PATH = Path(
    "artifacts/models/calibrated_test_metrics.json"
)

TARGET_COLUMN = "is_fraud"
TIMESTAMP_COLUMN = "transaction_timestamp"

# Matches the chronological calibration experiment.
VALIDATION_END_FRACTION = 0.85

# Selected on validation data and frozen before test evaluation.
OPERATING_THRESHOLD = 0.23

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_BINS = 10


def load_feature_columns() -> list[str]:
    """Load the ordered feature list saved with the calibration model.

    Returns:
        Model feature names in the exact order expected by the model.

    Raises:
        FileNotFoundError:
            If the feature-list artifact is missing.
        ValueError:
            If the saved list is invalid or contains leakage columns.
    """
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"Feature list not found: {FEATURE_LIST_PATH}"
        )

    with FEATURE_LIST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or not all(
            isinstance(column, str)
            for column in feature_columns
        )
    ):
        raise ValueError(
            "The feature list must be a non-empty list of strings."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            "The feature list contains duplicate column names."
        )

    forbidden_columns = {
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        "transaction_id",
    }

    leakage_columns = (
        set(feature_columns) & forbidden_columns
    )

    if leakage_columns:
        raise ValueError(
            f"Target or identifier leakage detected: {leakage_columns}"
        )

    return feature_columns


def load_test_split(
    feature_columns: list[str],
) -> pd.DataFrame:
    """Load the final chronological 15% of the feature dataset.

    The boundary matches the calibration experiment: training occupies the
    first 60%, calibration the next 10%, validation the next 15%, and the
    reserved test period the final 15%.

    Args:
        feature_columns: Ordered model feature names.

    Returns:
        A copy of the reserved test partition.

    Raises:
        FileNotFoundError:
            If the feature dataset does not exist.
        ValueError:
            If required data is missing or chronological ordering is invalid.
    """
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATASET_PATH}"
        )

    required_columns = [
        TIMESTAMP_COLUMN,
        *feature_columns,
        TARGET_COLUMN,
    ]

    df = pd.read_parquet(
        FEATURE_DATASET_PATH,
        columns=required_columns,
    )

    if df.empty:
        raise ValueError("The feature dataset is empty.")

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        utc=True,
    )

    if df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "The dataset contains missing transaction timestamps."
        )

    if not df[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError(
            "The dataset is not in chronological order. "
            "The time-based test split cannot be trusted."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("The target contains missing labels.")

    observed_labels = set(
        df[TARGET_COLUMN].unique().tolist()
    )

    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary fraud labels; got {observed_labels}."
        )

    total_rows = len(df)
    test_start = int(
        total_rows * VALIDATION_END_FRACTION
    )

    test_df = df.iloc[test_start:].copy()

    if test_df.empty:
        raise ValueError("The reserved test partition is empty.")

    if test_df[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "The test partition must contain both label classes."
        )

    print(f"Total dataset rows: {total_rows:,}")
    print(f"Test rows: {len(test_df):,}")
    print(
        f"Test period: {test_df[TIMESTAMP_COLUMN].iloc[0]} "
        f"to {test_df[TIMESTAMP_COLUMN].iloc[-1]}"
    )

    return test_df


def prepare_feature_matrix(
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Prepare the test features in the model's expected format.

    The first transaction for a customer can legitimately have a null
    seconds_since_previous_tx value. This baseline represents that case as
    zero. Any other missing features are treated as data-quality failures.

    Args:
        test_df: Reserved chronological test partition.
        feature_columns: Ordered model feature names.

    Returns:
        Finite float32 feature matrix.

    Raises:
        ValueError:
            If unexpected null or non-finite values are present.
    """
    X_test = test_df[feature_columns].copy()

    allowed_null_feature = "seconds_since_previous_tx"

    unexpected_nulls = {
        column: int(count)
        for column, count in X_test.isna().sum().items()
        if count > 0 and column != allowed_null_feature
    }

    if unexpected_nulls:
        raise ValueError(
            f"Unexpected missing test features: {unexpected_nulls}"
        )

    if allowed_null_feature in X_test.columns:
        X_test[allowed_null_feature] = (
            X_test[allowed_null_feature].fillna(0.0)
        )

    X_test = X_test.astype(np.float32)

    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError(
            "Test features contain non-finite numeric values."
        )

    return X_test


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = CALIBRATION_BINS,
) -> float:
    """Calculate an equally spaced-bin estimate of calibration error.

    Each bin contributes the absolute difference between its average
    predicted probability and its observed fraud rate, weighted by the
    fraction of observations in that bin.

    Lower values indicate closer agreement under this particular binning
    method. The estimate is a diagnostic rather than a complete assessment
    of calibration.

    Args:
        y_true: Binary fraud labels.
        probabilities: Predicted fraud probabilities.
        number_of_bins: Number of equally spaced probability bins.

    Returns:
        Expected calibration error as a float.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)

    edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    error = 0.0

    for bin_index in range(number_of_bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]

        if bin_index == number_of_bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)

        if not mask.any():
            continue

        observed_rate = float(y[mask].mean())
        mean_probability = float(p[mask].mean())
        bin_weight = float(mask.mean())

        error += (
            bin_weight
            * abs(observed_rate - mean_probability)
        )

    return float(error)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Calculate test probability-quality and threshold metrics.

    Args:
        y_true: Binary ground-truth fraud labels.
        probabilities: Sigmoid-calibrated fraud probabilities.

    Returns:
        A dictionary containing ranking, calibration, and classification
        metrics at the frozen operating threshold.

    Raises:
        ValueError:
            If predictions are invalid or their count differs from the labels.
    """
    y_true = np.asarray(y_true, dtype=np.uint8)
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Prediction count does not match the number of labels."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Predicted probabilities contain non-finite values."
        )

    if (
        (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError(
            "Predicted probabilities must be between zero and one."
        )

    predictions = (
        probabilities >= OPERATING_THRESHOLD
    ).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total_rows = len(y_true)
    fraud_count = int(y_true.sum())
    non_fraud_count = total_rows - fraud_count
    alerts = int(predictions.sum())

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    false_positive_rate = (
        float(fp / non_fraud_count)
        if non_fraud_count > 0
        else 0.0
    )

    fraud_capture_rate = (
        float(tp / fraud_count)
        if fraud_count > 0
        else 0.0
    )

    alert_rate = (
        float(alerts / total_rows)
        if total_rows > 0
        else 0.0
    )

    return {
        "rows": int(total_rows),
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "fraud_prevalence": float(fraud_count / total_rows),
        "mean_predicted_probability": float(probabilities.mean()),
        "threshold": OPERATING_THRESHOLD,
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "ece_10_bins": expected_calibration_error(
            y_true,
            probabilities,
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": false_positive_rate,
        "fraud_capture_rate": fraud_capture_rate,
        "alert_rate": alert_rate,
        "alerts": alerts,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print the calibrated model's test metrics in readable sections.

    Args:
        metrics: Test metrics returned by calculate_metrics.
    """
    print("\nProbability-quality metrics")
    print("-" * 45)
    print(f"PR-AUC:                  {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:                 {metrics['roc_auc']:.4f}")
    print(f"Brier score:             {metrics['brier_score']:.5f}")
    print(f"Log loss:                {metrics['log_loss']:.5f}")
    print(f"ECE (10 bins):           {metrics['ece_10_bins']:.5f}")
    print(
        f"Mean predicted probability: "
        f"{metrics['mean_predicted_probability']:.4%}"
    )
    print(
        f"Observed fraud prevalence: "
        f"{metrics['fraud_prevalence']:.4%}"
    )

    print("\nThreshold metrics")
    print("-" * 45)
    print(f"Frozen threshold:        {metrics['threshold']:.2f}")
    print(f"Precision:               {metrics['precision']:.4%}")
    print(f"Recall:                  {metrics['recall']:.4%}")
    print(f"F1:                      {metrics['f1']:.4f}")
    print(
        f"False-positive rate:     "
        f"{metrics['false_positive_rate']:.4%}"
    )
    print(
        f"Fraud capture rate:      "
        f"{metrics['fraud_capture_rate']:.4%}"
    )
    print(f"Alert rate:              {metrics['alert_rate']:.4%}")
    print(f"Alerts generated:        {metrics['alerts']:,}")

    print("\nConfusion matrix counts")
    print("-" * 45)
    print(f"True positives:          {metrics['true_positives']:,}")
    print(f"False positives:         {metrics['false_positives']:,}")
    print(f"False negatives:         {metrics['false_negatives']:,}")
    print(f"True negatives:           {metrics['true_negatives']:,}")


def main() -> None:
    """Run and save the final evaluation for the calibrated experiment."""
    print("Shepherd calibrated-model test evaluation")
    print("=" * 50)

    for artifact_path in (MODEL_PATH, CALIBRATORS_PATH):
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Required model artifact not found: {artifact_path}"
            )

    feature_columns = load_feature_columns()
    test_df = load_test_split(feature_columns)

    y_test = test_df[TARGET_COLUMN].to_numpy(
        dtype=np.uint8
    )

    print("\nLoading XGBoost model...")
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))

    print("Loading sigmoid calibrator...")
    calibrators = joblib.load(CALIBRATORS_PATH)

    if CALIBRATION_METHOD not in calibrators:
        raise ValueError(
            f"Calibrator '{CALIBRATION_METHOD}' is missing. "
            f"Available: {list(calibrators.keys())}"
        )

    calibrator = calibrators[CALIBRATION_METHOD]

    print("Preparing test features...")
    X_test = prepare_feature_matrix(
        test_df,
        feature_columns,
    )

    dtest = xgb.DMatrix(
        X_test,
        feature_names=feature_columns,
    )

    del X_test
    gc.collect()

    print("Generating model scores...")
    raw_margins = model.predict(
        dtest,
        output_margin=True,
    )

    del dtest
    del model
    gc.collect()

    print("Applying sigmoid calibration...")
    probabilities = calibrator.predict_proba(
        raw_margins.reshape(-1, 1)
    )[:, 1]

    del raw_margins
    del calibrator
    gc.collect()

    print("Calculating metrics at frozen threshold 0.23...")
    metrics = calculate_metrics(
        y_true=y_test,
        probabilities=probabilities,
    )

    print_metrics(metrics)

    results = {
        "experiment_name": "shepherd_calibrated_model_test_evaluation",
        "model_artifact": str(MODEL_PATH),
        "calibrator_artifact": str(CALIBRATORS_PATH),
        "feature_dataset": str(FEATURE_DATASET_PATH),
        "calibration_method": CALIBRATION_METHOD,
        "threshold_selection": {
            "threshold": OPERATING_THRESHOLD,
            "selected_using": "validation partition",
            "policy": (
                "Maximize recall subject to at least 50% precision "
                "on validation data."
            ),
        },
        "test_metrics": metrics,
        "notes": [
            "The test set was not used to choose the operating threshold.",
            "No threshold optimization was performed on test labels.",
            "The test period was previously examined for the original baseline "
            "experiment, so it is not completely untouched across the project.",
            "Results reflect a synthetic fraud environment and should not be "
            "represented as evidence of real-world fraud detection efficacy.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(f"\nSaved test evaluation: {OUTPUT_PATH}")
    print("Calibrated test evaluation completed.")


if __name__ == "__main__":
    main()
