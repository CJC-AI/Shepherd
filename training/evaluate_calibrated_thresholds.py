
"""Select operating thresholds for Shepherd's sigmoid-calibrated model.

This module evaluates thresholds using the chronological validation partition
from the probability-calibration experiment.

It loads the calibration-experiment XGBoost model and the sigmoid calibrator,
then generates calibrated fraud probabilities for validation transactions.

Threshold selection follows this operational rule:

    Choose the threshold with the highest validation recall while achieving
    at least a specified minimum precision.

The script evaluates minimum precision targets of 50%, 60%, and 70%.
It searches thresholds from 0.01 through 0.99 in increments of 0.01.

All threshold selection is performed on validation data. The final
chronological test partition is not used to select or compare thresholds.

The script saves the full threshold evaluation table and a JSON summary
of the recommended validation operating points. It does not overwrite
the original baseline model, calibration model, or calibrators.

Usage:
    python -m training.evaluate_calibrated_thresholds
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

OUTPUT_DIRECTORY = Path("artifacts/models")

THRESHOLD_TABLE_PATH = (
    OUTPUT_DIRECTORY / "calibrated_threshold_metrics.csv"
)

RECOMMENDATIONS_PATH = (
    OUTPUT_DIRECTORY / "calibrated_threshold_recommendations.json"
)

TARGET_COLUMN = "is_fraud"
TIMESTAMP_COLUMN = "transaction_timestamp"

# These boundaries must match the calibration experiment.
TRAIN_FRACTION = 0.60
CALIBRATION_END_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

# The chosen calibration method from the preceding experiment.
CALIBRATION_METHOD = "sigmoid"

# Business operating targets to compare on the validation set.
MINIMUM_PRECISION_TARGETS = (0.50, 0.60, 0.70)

# Evaluate 99 candidate thresholds: 0.01, 0.02, ..., 0.99.
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100.0


def load_feature_columns() -> list[str]:
    """Load and validate the feature order saved with the model.

    Returns:
        Ordered model feature names.

    Raises:
        FileNotFoundError:
            If the feature-list artifact is missing.
        ValueError:
            If the artifact does not contain a valid non-empty list.
    """
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"Model feature list not found: {FEATURE_LIST_PATH}"
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
            "The saved model feature list must be a non-empty "
            "list of column names."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            "The saved model feature list contains duplicates."
        )

    if TARGET_COLUMN in feature_columns:
        raise ValueError(
            "The fraud target must not be included as a model feature."
        )

    if TIMESTAMP_COLUMN in feature_columns:
        raise ValueError(
            "The transaction timestamp must not be included as a model feature."
        )

    return feature_columns


def load_validation_split(
    feature_columns: list[str],
) -> pd.DataFrame:
    """Load the chronological validation partition.

    The partition boundaries match those used by the calibration experiment:
    60% training, 10% calibration, 15% validation, and 15% reserved test.

    Args:
        feature_columns: Ordered feature columns expected by the model.

    Returns:
        A DataFrame containing validation features, timestamps, and labels.

    Raises:
        FileNotFoundError:
            If the feature dataset is missing.
        ValueError:
            If required columns or valid chronological ordering are absent.
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
            "Transaction timestamps contain missing values."
        )

    if not df[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError(
            "The feature dataset is not chronologically ordered."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Fraud labels contain missing values."
        )

    observed_labels = set(
        df[TARGET_COLUMN].unique().tolist()
    )

    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary fraud labels, got {observed_labels}."
        )

    total_rows = len(df)

    calibration_end = int(
        total_rows * CALIBRATION_END_FRACTION
    )

    validation_end = int(
        total_rows * VALIDATION_END_FRACTION
    )

    validation_df = df.iloc[
        calibration_end:validation_end
    ].copy()

    if validation_df.empty:
        raise ValueError(
            "The chronological validation partition is empty."
        )

    if validation_df[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "Validation data must contain both fraud and non-fraud examples."
        )

    print(f"Total dataset rows: {total_rows:,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(
        "Validation period: "
        f"{validation_df[TIMESTAMP_COLUMN].iloc[0]} "
        "to "
        f"{validation_df[TIMESTAMP_COLUMN].iloc[-1]}"
    )

    return validation_df


def prepare_feature_matrix(
    validation_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Prepare validation features in the saved model's expected order.

    The known nullable feature seconds_since_previous_tx is filled with zero
    for transactions that have no previous transaction. Other unexpected
    missing or non-finite feature values raise an error.

    Args:
        validation_df: Chronological validation partition.
        feature_columns: Ordered model feature names.

    Returns:
        A finite float32 feature matrix.

    Raises:
        ValueError:
            If unexpected missing or non-finite values are found.
    """
    X_validation = validation_df[feature_columns].copy()

    allowed_null_feature = "seconds_since_previous_tx"

    missing_counts = X_validation.isna().sum()

    unexpected_missing = {
        column: int(count)
        for column, count in missing_counts.items()
        if count > 0 and column != allowed_null_feature
    }

    if unexpected_missing:
        raise ValueError(
            f"Unexpected missing validation features: "
            f"{unexpected_missing}"
        )

    if allowed_null_feature in X_validation.columns:
        X_validation[allowed_null_feature] = (
            X_validation[allowed_null_feature].fillna(0.0)
        )

    X_validation = X_validation.astype(np.float32)

    if not np.isfinite(X_validation.to_numpy()).all():
        raise ValueError(
            "Validation features contain non-finite values."
        )

    return X_validation


def load_calibrated_probabilities(
    validation_df: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Generate sigmoid-calibrated probabilities for validation rows.

    The sigmoid calibrator was fitted on the separate calibration period.
    Validation labels are not used to fit the model or calibrator.

    Args:
        validation_df: Chronological validation partition.
        feature_columns: Ordered model feature names.

    Returns:
        Array of calibrated fraud probabilities.

    Raises:
        FileNotFoundError:
            If the model or calibrator artifact is missing.
        ValueError:
            If artifacts are incomplete or predictions are invalid.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Calibration-experiment model not found: {MODEL_PATH}"
        )

    if not CALIBRATORS_PATH.exists():
        raise FileNotFoundError(
            f"Calibrator artifact not found: {CALIBRATORS_PATH}"
        )

    print("\nLoading calibration-experiment model...")

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))

    print("Loading fitted calibrators...")
    calibrators = joblib.load(CALIBRATORS_PATH)

    if CALIBRATION_METHOD not in calibrators:
        raise ValueError(
            f"Calibrator '{CALIBRATION_METHOD}' not found. "
            f"Available calibrators: {list(calibrators.keys())}"
        )

    sigmoid_calibrator = calibrators[CALIBRATION_METHOD]

    print("Preparing validation features...")
    X_validation = prepare_feature_matrix(
        validation_df,
        feature_columns,
    )

    dvalidation = xgb.DMatrix(
        X_validation,
        feature_names=feature_columns,
    )

    del X_validation
    gc.collect()

    print("Predicting raw model margins...")
    raw_margins = model.predict(
        dvalidation,
        output_margin=True,
    )

    del dvalidation
    del model
    gc.collect()

    print("Applying sigmoid calibration...")
    calibrated_probabilities = sigmoid_calibrator.predict_proba(
        raw_margins.reshape(-1, 1)
    )[:, 1]

    if not np.isfinite(calibrated_probabilities).all():
        raise ValueError(
            "Calibrated predictions contain non-finite values."
        )

    if (
        (calibrated_probabilities < 0.0).any()
        or (calibrated_probabilities > 1.0).any()
    ):
        raise ValueError(
            "Calibrated probabilities must lie between zero and one."
        )

    return calibrated_probabilities


def evaluate_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate operating metrics for one probability threshold.

    Args:
        y_true: Binary ground-truth fraud labels.
        probabilities: Sigmoid-calibrated fraud probabilities.
        threshold: Probability cutoff for producing a fraud alert.

    Returns:
        A dictionary of classification and operational metrics.
    """
    y_true = np.asarray(y_true, dtype=np.uint8)
    predictions = (probabilities >= threshold).astype(np.uint8)

    true_positives = int(
        np.sum((y_true == 1) & (predictions == 1))
    )

    false_positives = int(
        np.sum((y_true == 0) & (predictions == 1))
    )

    false_negatives = int(
        np.sum((y_true == 1) & (predictions == 0))
    )

    true_negatives = int(
        np.sum((y_true == 0) & (predictions == 0))
    )

    total_rows = len(y_true)
    fraud_count = int(y_true.sum())
    non_fraud_count = total_rows - fraud_count
    alerts = true_positives + false_positives

    precision = (
        true_positives / alerts
        if alerts > 0
        else 0.0
    )

    recall = (
        true_positives / fraud_count
        if fraud_count > 0
        else 0.0
    )

    f1_denominator = precision + recall

    f1 = (
        2.0 * precision * recall / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    false_positive_rate = (
        false_positives / non_fraud_count
        if non_fraud_count > 0
        else 0.0
    )

    alert_rate = (
        alerts / total_rows
        if total_rows > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "fraud_capture_rate": float(recall),
        "alert_rate": float(alert_rate),
        "alerts": int(alerts),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_negatives": int(true_negatives),
    }


def evaluate_threshold_grid(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all configured probability thresholds.

    Args:
        y_true: Binary ground-truth fraud labels.
        probabilities: Sigmoid-calibrated fraud probabilities.

    Returns:
        A DataFrame with one row for each tested threshold.
    """
    rows = [
        evaluate_threshold(
            y_true=y_true,
            probabilities=probabilities,
            threshold=float(threshold),
        )
        for threshold in THRESHOLDS
    ]

    return pd.DataFrame(rows)


def select_threshold_for_precision(
    results: pd.DataFrame,
    minimum_precision: float,
) -> dict[str, Any] | None:
    """Select the highest-recall threshold meeting a precision target.

    Among thresholds whose measured validation precision meets the requested
    minimum, the function prioritizes recall, then precision, then the lower
    threshold when the other two values are tied.

    Args:
        results: Threshold evaluation table.
        minimum_precision: Required minimum validation precision, from zero
            through one.

    Returns:
        The selected metrics as a JSON-compatible dictionary, or None if no
        tested threshold meets the requested precision.
    """
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError(
            "Minimum precision must be between zero and one."
        )

    eligible = results.loc[
        results["precision"] >= minimum_precision
    ].copy()

    if eligible.empty:
        return None

    best_row = eligible.sort_values(
        by=["recall", "precision", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]

    return {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in best_row.to_dict().items()
    }


def print_recommendations(
    recommendations: dict[str, dict[str, Any] | None],
) -> None:
    """Print threshold recommendations for the configured precision goals.

    Args:
        recommendations: Mapping from precision target labels to selected
            threshold metrics, or None when no threshold meets the target.
    """
    print("\nRecommended validation operating points")
    print("=" * 105)
    print(
        f"{'Min precision':<16}"
        f"{'Threshold':>11}"
        f"{'Precision':>12}"
        f"{'Recall':>11}"
        f"{'F1':>10}"
        f"{'Alerts':>12}"
        f"{'Alert rate':>13}"
        f"{'False-pos rate':>17}"
    )
    print("-" * 105)

    for target_label, selected in recommendations.items():
        if selected is None:
            print(
                f"{target_label:<16}"
                "No threshold in the configured grid met this target."
            )
            continue

        print(
            f"{target_label:<16}"
            f"{selected['threshold']:>11.2f}"
            f"{selected['precision']:>11.2%}"
            f"{selected['recall']:>10.2%}"
            f"{selected['f1']:>10.4f}"
            f"{selected['alerts']:>12,}"
            f"{selected['alert_rate']:>12.2%}"
            f"{selected['false_positive_rate']:>16.2%}"
        )

    print("=" * 105)
    print(
        "Selection rule: maximize validation recall while satisfying "
        "the minimum precision target."
    )


def main() -> None:
    """Run calibrated threshold selection using validation data only."""
    print("Shepherd calibrated-threshold evaluation")
    print("=" * 50)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_columns = load_feature_columns()

    validation_df = load_validation_split(feature_columns)

    y_validation = validation_df[TARGET_COLUMN].to_numpy(
        dtype=np.uint8
    )

    calibrated_probabilities = load_calibrated_probabilities(
        validation_df=validation_df,
        feature_columns=feature_columns,
    )

    if len(calibrated_probabilities) != len(y_validation):
        raise ValueError(
            "Prediction count does not match validation label count."
        )

    print("\nCalibrated validation score summary")
    print("-" * 50)
    print(f"Fraud prevalence: {y_validation.mean():.4%}")
    print(
        f"Mean calibrated probability: "
        f"{calibrated_probabilities.mean():.4%}"
    )
    print(
        f"Minimum calibrated probability: "
        f"{calibrated_probabilities.min():.6f}"
    )
    print(
        f"Maximum calibrated probability: "
        f"{calibrated_probabilities.max():.6f}"
    )
    print(
        f"Validation PR-AUC: "
        f"{average_precision_score(y_validation, calibrated_probabilities):.4f}"
    )
    print(
        f"Validation ROC-AUC: "
        f"{roc_auc_score(y_validation, calibrated_probabilities):.4f}"
    )

    print("\nEvaluating threshold grid...")
    results = evaluate_threshold_grid(
        y_true=y_validation,
        probabilities=calibrated_probabilities,
    )

    results.to_csv(
        THRESHOLD_TABLE_PATH,
        index=False,
    )

    print(f"Saved full threshold table: {THRESHOLD_TABLE_PATH}")

    recommendations: dict[str, dict[str, Any] | None] = {}

    for minimum_precision in MINIMUM_PRECISION_TARGETS:
        target_label = f"{minimum_precision:.0%}"
        recommendations[target_label] = (
            select_threshold_for_precision(
                results=results,
                minimum_precision=minimum_precision,
            )
        )

    print_recommendations(recommendations)

    summary = {
        "experiment_name": (
            "shepherd_sigmoid_calibrated_threshold_selection"
        ),
        "calibration_method": CALIBRATION_METHOD,
        "model_artifact": str(MODEL_PATH),
        "calibrator_artifact": str(CALIBRATORS_PATH),
        "feature_dataset": str(FEATURE_DATASET_PATH),
        "threshold_table_artifact": str(THRESHOLD_TABLE_PATH),
        "validation_rows": int(len(validation_df)),
        "validation_fraud_count": int(y_validation.sum()),
        "validation_fraud_rate": float(y_validation.mean()),
        "validation_pr_auc": float(
            average_precision_score(
                y_validation,
                calibrated_probabilities,
            )
        ),
        "validation_roc_auc": float(
            roc_auc_score(
                y_validation,
                calibrated_probabilities,
            )
        ),
        "threshold_grid": [
            float(threshold)
            for threshold in THRESHOLDS
        ],
        "selection_rule": (
            "Maximize validation recall subject to the specified "
            "minimum validation precision."
        ),
        "recommendations": recommendations,
        "notes": [
            "The sigmoid calibrator was fitted on the separate calibration period.",
            "Threshold selection used validation labels only.",
            "The reserved test partition was not used for this evaluation.",
            "The recommended thresholds are validation candidates, not yet "
            "final production decisions.",
            "No existing baseline or calibration artifacts were overwritten.",
        ],
    }

    with RECOMMENDATIONS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"\nSaved threshold recommendations: "
        f"{RECOMMENDATIONS_PATH}"
    )
    print("Calibrated threshold evaluation completed.")


if __name__ == "__main__":
    main()
