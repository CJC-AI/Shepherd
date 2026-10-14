
"""Train and calibrate a chronological Shepherd fraud detection model.

This module creates a separate calibration experiment without overwriting the
existing baseline model.

The dataset is partitioned chronologically as follows:

    First 60%  -> Model training
    Next 10%   -> Probability calibrator fitting
    Next 15%   -> Calibration-method evaluation
    Final 15%  -> Held-out test period, reserved for later evaluation

Two calibration methods are fitted:

    1. Sigmoid calibration (Platt scaling) using XGBoost raw margins.
    2. Isotonic regression using XGBoost predicted probabilities.

The two methods are compared on the validation period using Brier score,
log loss, expected calibration error, PR-AUC, and ROC-AUC.

Lower Brier score, log loss, and calibration error indicate better probability
quality. PR-AUC and ROC-AUC assess ranking quality rather than calibration.

The experiment saves a new XGBoost model, calibration objects, feature list,
and metadata. It does not overwrite the existing baseline artifacts.

Usage:
    python -m training.calibrate_model
"""

import gc
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from training.model_config import XGBoostConfig


FEATURE_DATASET_PATH = Path(
    "artifacts/features/transaction_features.parquet"
)

OUTPUT_DIRECTORY = Path("artifacts/models")

MODEL_PATH = OUTPUT_DIRECTORY / "xgboost_calibration_base.json"
CALIBRATORS_PATH = OUTPUT_DIRECTORY / "calibrators.joblib"
FEATURE_LIST_PATH = (
    OUTPUT_DIRECTORY / "xgboost_calibration_base_features.json"
)
METADATA_PATH = (
    OUTPUT_DIRECTORY / "calibration_experiment_metadata.json"
)

TARGET_COLUMN = "is_fraud"
TIMESTAMP_COLUMN = "transaction_timestamp"

# Chronological split boundaries.
TRAIN_FRACTION = 0.60
CALIBRATION_END_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

# The original baseline's best_iteration was 373.
# XGBoost iteration indices start at zero, so this corresponds to 374 rounds.
BOOST_ROUNDS = 374

RANDOM_STATE = 42
CALIBRATION_BINS = 10

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


def load_feature_dataset() -> pd.DataFrame:
    """Load the required feature columns and verify chronological order.

    Returns:
        A DataFrame containing model features, transaction timestamps,
        and fraud labels in chronological order.

    Raises:
        FileNotFoundError:
            If the feature dataset does not exist.
        ValueError:
            If required columns are missing, the dataset is empty,
            labels are invalid, or rows are not chronologically ordered.
    """
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATASET_PATH}"
        )

    required_columns = [
        TIMESTAMP_COLUMN,
        *MODEL_FEATURES,
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
            "The feature dataset is not chronologically ordered. "
            "Do not create time-based splits until ordering is corrected."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "The target column contains missing fraud labels."
        )

    observed_labels = set(
        df[TARGET_COLUMN].unique().tolist()
    )

    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary fraud labels 0 and 1; got {observed_labels}."
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.uint8)

    print(f"Loaded rows: {len(df):,}")
    print(
        f"Dataset period: {df[TIMESTAMP_COLUMN].min()} "
        f"to {df[TIMESTAMP_COLUMN].max()}"
    )

    return df


def prepare_feature_matrix(
    frame: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Prepare a numeric, finite feature matrix for XGBoost.

    The only expected feature nulls are in seconds_since_previous_tx,
    which is undefined for a customer's first transaction. For this
    baseline experiment, those nulls are represented as zero.

    Args:
        frame: A chronological dataset partition.
        split_name: A descriptive partition name for error messages.

    Returns:
        A float32 feature matrix with the expected model columns.

    Raises:
        ValueError:
            If unexpected missing or non-finite feature values are found.
    """
    X = frame[MODEL_FEATURES].copy()

    allowed_null_feature = "seconds_since_previous_tx"

    unexpected_nulls = {
        column: int(count)
        for column, count in X.isna().sum().items()
        if count > 0 and column != allowed_null_feature
    }

    if unexpected_nulls:
        raise ValueError(
            f"Unexpected missing features in {split_name}: "
            f"{unexpected_nulls}"
        )

    X[allowed_null_feature] = X[allowed_null_feature].fillna(0.0)
    X = X.astype(np.float32)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            f"Non-finite feature values detected in {split_name}."
        )

    return X


def describe_partition(
    name: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize a chronological partition for experiment metadata.

    Args:
        name: Name of the partition.
        frame: DataFrame containing timestamps and fraud labels.

    Returns:
        A JSON-compatible dictionary with row counts, dates, and fraud rate.
    """
    return {
        "name": name,
        "rows": int(len(frame)),
        "fraud_count": int(frame[TARGET_COLUMN].sum()),
        "fraud_rate": float(frame[TARGET_COLUMN].mean()),
        "first_timestamp": str(frame[TIMESTAMP_COLUMN].iloc[0]),
        "last_timestamp": str(frame[TIMESTAMP_COLUMN].iloc[-1]),
    }


def split_chronologically(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create training, calibration, validation, and test partitions.

    The first 60% is used to fit the model. The following 10% is reserved
    for calibrator fitting. Validation occupies the next 15%, and the final
    15% remains reserved for a later held-out evaluation.

    Args:
        df: Chronologically ordered feature dataset.

    Returns:
        A tuple containing training, calibration, validation, and test
        DataFrames, in that order.

    Raises:
        ValueError:
            If the dataset is too small to create the requested partitions.
    """
    total_rows = len(df)

    train_end = int(total_rows * TRAIN_FRACTION)
    calibration_end = int(
        total_rows * CALIBRATION_END_FRACTION
    )
    validation_end = int(
        total_rows * VALIDATION_END_FRACTION
    )

    train_df = df.iloc[:train_end]
    calibration_df = df.iloc[train_end:calibration_end]
    validation_df = df.iloc[calibration_end:validation_end]
    test_df = df.iloc[validation_end:]

    partitions = {
        "training": train_df,
        "calibration": calibration_df,
        "validation": validation_df,
        "test": test_df,
    }

    empty_partitions = [
        name for name, partition in partitions.items()
        if partition.empty
    ]

    if empty_partitions:
        raise ValueError(
            f"Empty chronological partitions: {empty_partitions}"
        )

    return (
        train_df,
        calibration_df,
        validation_df,
        test_df,
    )


def train_base_model(
    train_df: pd.DataFrame,
) -> xgb.Booster:
    """Train an XGBoost model using only the training period.

    The positive-class weight is derived from the training labels.
    The model is trained for a fixed number of boosting rounds carried
    forward from the earlier baseline experiment.

    Args:
        train_df: The first chronological training partition.

    Returns:
        A trained XGBoost Booster.

    Raises:
        ValueError:
            If the training partition does not contain both label classes.
    """
    y_train = train_df[TARGET_COLUMN].astype(np.uint8).to_numpy()

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "The training partition must contain both fraud and "
            "non-fraud examples."
        )

    scale_pos_weight = negative_count / positive_count

    print("\nPreparing training features...")
    X_train = prepare_feature_matrix(train_df, "training")

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=MODEL_FEATURES,
    )

    del X_train
    gc.collect()

    config = XGBoostConfig()

    thread_count = max(1, (os.cpu_count() or 2) - 1)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": config.learning_rate,
        "max_depth": config.max_depth,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "reg_alpha": config.reg_alpha,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "seed": RANDOM_STATE,
        "nthread": thread_count,
        "verbosity": 0,
    }

    print(f"Training rows: {len(train_df):,}")
    print(f"Training fraud rate: {y_train.mean():.4%}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"Boosting rounds: {BOOST_ROUNDS}")
    print(f"CPU threads: {thread_count}")
    print("\nTraining calibration-experiment model...")

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=BOOST_ROUNDS,
        verbose_eval=False,
    )

    del dtrain
    gc.collect()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(str(MODEL_PATH))

    with FEATURE_LIST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            MODEL_FEATURES,
            file,
            indent=2,
        )

    print(f"Saved calibration-experiment model: {MODEL_PATH}")

    return model


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = CALIBRATION_BINS,
) -> float:
    """Estimate expected calibration error using equally spaced bins.

    For each probability bin, the absolute difference between mean predicted
    probability and observed fraud frequency is weighted by the fraction of
    observations in that bin.

    This is a useful diagnostic, but its value depends on the binning scheme.

    Args:
        y_true: Binary ground-truth fraud labels.
        probabilities: Predicted fraud probabilities.
        number_of_bins: Number of equally spaced probability bins.

    Returns:
        A float representing the estimated expected calibration error.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)

    edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    total_error = 0.0

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

        total_error += (
            bin_weight * abs(observed_rate - mean_probability)
        )

    return float(total_error)


def evaluate_probability_quality(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate ranking and probability-calibration metrics.

    Args:
        y_true: Binary ground-truth fraud labels.
        probabilities: Predicted fraud probabilities.

    Returns:
        A dictionary containing PR-AUC, ROC-AUC, Brier score, log loss,
        and expected calibration error.
    """
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    return {
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
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
    }


def fit_calibrators(
    model: xgb.Booster,
    calibration_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fit sigmoid and isotonic calibrators on the calibration period.

    Sigmoid calibration is fitted on the model's raw margins, while isotonic
    regression is fitted on the model's ordinary probability predictions.
    Neither calibrator is trained on the validation or test labels.

    Args:
        model: Trained XGBoost Booster.
        calibration_df: Chronological calibration partition.

    Returns:
        A dictionary containing the fitted calibration objects.

    Raises:
        ValueError:
            If the calibration partition has only one label class.
    """
    y_calibration = calibration_df[TARGET_COLUMN].astype(
        np.uint8
    ).to_numpy()

    if len(np.unique(y_calibration)) != 2:
        raise ValueError(
            "The calibration period must contain both fraud and "
            "non-fraud examples."
        )

    print("\nPreparing calibration features...")
    X_calibration = prepare_feature_matrix(
        calibration_df,
        "calibration",
    )

    dcalibration = xgb.DMatrix(
        X_calibration,
        feature_names=MODEL_FEATURES,
    )

    del X_calibration
    gc.collect()

    raw_margins = model.predict(
        dcalibration,
        output_margin=True,
    )

    raw_probabilities = model.predict(dcalibration)

    del dcalibration
    gc.collect()

    print(
        f"Calibration rows: {len(calibration_df):,}"
    )
    print(
        f"Calibration fraud rate: {y_calibration.mean():.4%}"
    )
    print("Fitting sigmoid calibration...")

    sigmoid_calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    sigmoid_calibrator.fit(
        raw_margins.reshape(-1, 1),
        y_calibration,
    )

    print("Fitting isotonic calibration...")

    isotonic_calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic_calibrator.fit(
        raw_probabilities,
        y_calibration,
    )

    calibrators = {
        "sigmoid": sigmoid_calibrator,
        "isotonic": isotonic_calibrator,
    }

    joblib.dump(
        calibrators,
        CALIBRATORS_PATH,
    )

    print(f"Saved calibrators: {CALIBRATORS_PATH}")

    return calibrators


def compare_calibrators_on_validation(
    model: xgb.Booster,
    calibrators: dict[str, Any],
    validation_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compare raw and calibrated probabilities on validation data.

    The validation period is not used to fit either calibration model.
    These results will help determine which calibration method to carry
    forward into the next threshold-selection stage.

    Args:
        model: Trained XGBoost Booster.
        calibrators: Fitted sigmoid and isotonic calibrators.
        validation_df: Later chronological validation partition.

    Returns:
        A dictionary mapping each method to its probability-quality metrics.
    """
    y_validation = validation_df[TARGET_COLUMN].astype(
        np.uint8
    ).to_numpy()

    print("\nPreparing validation features...")
    X_validation = prepare_feature_matrix(
        validation_df,
        "validation",
    )

    dvalidation = xgb.DMatrix(
        X_validation,
        feature_names=MODEL_FEATURES,
    )

    del X_validation
    gc.collect()

    raw_margins = model.predict(
        dvalidation,
        output_margin=True,
    )

    raw_probabilities = model.predict(dvalidation)

    del dvalidation
    gc.collect()

    sigmoid_probabilities = calibrators[
        "sigmoid"
    ].predict_proba(
        raw_margins.reshape(-1, 1)
    )[:, 1]

    isotonic_probabilities = calibrators[
        "isotonic"
    ].predict(raw_probabilities)

    results = {
        "raw": evaluate_probability_quality(
            y_validation,
            raw_probabilities,
        ),
        "sigmoid": evaluate_probability_quality(
            y_validation,
            sigmoid_probabilities,
        ),
        "isotonic": evaluate_probability_quality(
            y_validation,
            isotonic_probabilities,
        ),
    }

    return results


def print_comparison(
    results: dict[str, dict[str, float]],
) -> None:
    """Print validation metrics for raw and calibrated probabilities.

    Args:
        results: Dictionary returned by the validation comparison function.
    """
    print("\nValidation probability-quality comparison")
    print("=" * 93)
    print(
        f"{'Method':<12}"
        f"{'PR-AUC':>10}"
        f"{'ROC-AUC':>10}"
        f"{'Brier':>12}"
        f"{'Log loss':>12}"
        f"{'ECE (10 bins)':>18}"
    )
    print("-" * 93)

    for method, metrics in results.items():
        print(
            f"{method:<12}"
            f"{metrics['pr_auc']:>10.4f}"
            f"{metrics['roc_auc']:>10.4f}"
            f"{metrics['brier_score']:>12.5f}"
            f"{metrics['log_loss']:>12.5f}"
            f"{metrics['ece_10_bins']:>18.5f}"
        )

    print("-" * 93)
    print("Lower Brier, log loss, and ECE are generally better.")
    print("Higher PR-AUC and ROC-AUC indicate stronger ranking.")


def main() -> None:
    """Run the full chronological calibration experiment."""
    print("Shepherd probability-calibration experiment")
    print("=" * 50)

    df = load_feature_dataset()

    (
        train_df,
        calibration_df,
        validation_df,
        test_df,
    ) = split_chronologically(df)

    partition_summaries = {
        "training": describe_partition("training", train_df),
        "calibration": describe_partition(
            "calibration",
            calibration_df,
        ),
        "validation": describe_partition(
            "validation",
            validation_df,
        ),
        "test_reserved": describe_partition("test", test_df),
    }

    print("\nChronological partitions")
    print("-" * 50)

    for name, summary in partition_summaries.items():
        print(
            f"{name:<16}"
            f"rows={summary['rows']:>9,}  "
            f"fraud_rate={summary['fraud_rate']:.4%}"
        )

    # Train and calibrate without fitting on validation or test labels.
    model = train_base_model(train_df)

    calibrators = fit_calibrators(
        model=model,
        calibration_df=calibration_df,
    )

    results = compare_calibrators_on_validation(
        model=model,
        calibrators=calibrators,
        validation_df=validation_df,
    )

    print_comparison(results)

    metadata = {
        "experiment_name": "shepherd_probability_calibration",
        "feature_dataset": str(FEATURE_DATASET_PATH),
        "model_artifact": str(MODEL_PATH),
        "calibrator_artifact": str(CALIBRATORS_PATH),
        "feature_list_artifact": str(FEATURE_LIST_PATH),
        "model_features": MODEL_FEATURES,
        "split_fractions": {
            "training": TRAIN_FRACTION,
            "calibration": (
                CALIBRATION_END_FRACTION - TRAIN_FRACTION
            ),
            "validation": (
                VALIDATION_END_FRACTION
                - CALIBRATION_END_FRACTION
            ),
            "test": 1.0 - VALIDATION_END_FRACTION,
        },
        "boost_rounds": BOOST_ROUNDS,
        "random_state": RANDOM_STATE,
        "partitions": partition_summaries,
        "validation_metrics": results,
        "notes": [
            "The existing baseline artifacts were not overwritten.",
            "Sigmoid calibration uses XGBoost raw margins.",
            "Isotonic calibration uses XGBoost probabilities.",
            "Validation labels were not used to fit the calibrators.",
            "The final chronological test period was reserved for later "
            "evaluation and was not scored by this script.",
            "The test period has previously been examined for the original "
            "baseline, so it should not be described as completely unseen "
            "at the overall project level.",
        ],
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(f"\nSaved experiment metadata: {METADATA_PATH}")
    print("Calibration experiment completed.")
    print(
        "No operating threshold has been selected or changed by this script."
    )


if __name__ == "__main__":
    main()
