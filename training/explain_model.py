"""Explain Shepherd's XGBoost fraud model using SHAP.

This module explains the XGBoost model from the probability-calibration
experiment. It uses a reproducible sample from the chronological validation
period and does not use the reserved test period.

The script generates:
    - Global mean absolute SHAP feature importance.
    - A SHAP beeswarm summary plot.
    - A SHAP global bar plot.
    - A SHAP dependence plot for tx_count_10m.
    - Feature-level explanations for ten high-scoring validation transactions.
    - Metadata describing the explanation experiment.

SHAP values explain the underlying XGBoost raw margin (log-odds). Positive
SHAP values push the model score toward the fraud class; negative values push
the score away from the fraud class. They are not direct additive
contributions to the final sigmoid-calibrated probability.

The local explanations are generated for the ten validation transactions
with the highest calibrated fraud probabilities in the sampled dataset.
These examples illustrate model behavior; they do not establish why a real
transaction is fraudulent.

Usage:
    python -m training.explain_model
"""

import gc
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


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

OUTPUT_DIRECTORY = Path("artifacts/explainability")

TARGET_COLUMN = "is_fraud"
TIMESTAMP_COLUMN = "transaction_timestamp"
TRANSACTION_ID_COLUMN = "transaction_id"

# These boundaries match the calibration experiment.
VALIDATION_START_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

CALIBRATION_METHOD = "sigmoid"

# Keep SHAP computation manageable and reproducible.
SHAP_SAMPLE_SIZE = 5_000
LOCAL_EXAMPLE_COUNT = 10
RANDOM_STATE = 42


def load_feature_columns() -> list[str]:
    """Load the ordered model feature list.

    Returns:
        The list of feature names in the order expected by XGBoost.

    Raises:
        FileNotFoundError:
            If the saved feature list is missing.
        ValueError:
            If the saved feature list is invalid or contains leakage columns.
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
            "The feature list contains duplicate features."
        )

    forbidden_columns = {
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        TRANSACTION_ID_COLUMN,
    }

    leakage_columns = set(feature_columns) & forbidden_columns

    if leakage_columns:
        raise ValueError(
            f"Target or identifier leakage detected: {leakage_columns}"
        )

    return feature_columns


def load_validation_sample(
    feature_columns: list[str],
) -> pd.DataFrame:
    """Load a reproducible sample from the validation period.

    The validation period is the section after the first 70% and before
    the first 85% of chronologically ordered transactions.

    Args:
        feature_columns: Ordered model feature names.

    Returns:
        A sampled DataFrame containing identifiers, timestamps, features,
        and labels.

    Raises:
        FileNotFoundError:
            If the feature dataset does not exist.
        ValueError:
            If required fields are missing or chronological ordering fails.
    """
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATASET_PATH}"
        )

    required_columns = [
        TRANSACTION_ID_COLUMN,
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
            "The dataset is not chronologically ordered."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Fraud labels contain missing values."
        )

    total_rows = len(df)

    validation_start = int(
        total_rows * VALIDATION_START_FRACTION
    )

    validation_end = int(
        total_rows * VALIDATION_END_FRACTION
    )

    validation_df = df.iloc[
        validation_start:validation_end
    ]

    if validation_df.empty:
        raise ValueError("The validation partition is empty.")

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(validation_df),
    )

    sample_df = validation_df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(f"Full dataset rows: {total_rows:,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"SHAP sample rows: {len(sample_df):,}")
    print(
        "Sampled validation period: "
        f"{sample_df[TIMESTAMP_COLUMN].min()} to "
        f"{sample_df[TIMESTAMP_COLUMN].max()}"
    )

    return sample_df


def prepare_feature_matrix(
    sample_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Prepare finite numeric features in the model's expected order.

    The null value in seconds_since_previous_tx is expected for a customer's
    first transaction. This experiment uses the same zero substitution as
    the training and evaluation scripts.

    Args:
        sample_df: Sampled validation transactions.
        feature_columns: Ordered model feature names.

    Returns:
        A float32 feature matrix.

    Raises:
        ValueError:
            If unexpected missing or non-finite feature values are found.
    """
    X = sample_df[feature_columns].copy()

    allowed_null_feature = "seconds_since_previous_tx"

    unexpected_nulls = {
        column: int(count)
        for column, count in X.isna().sum().items()
        if count > 0 and column != allowed_null_feature
    }

    if unexpected_nulls:
        raise ValueError(
            f"Unexpected missing feature values: {unexpected_nulls}"
        )

    if allowed_null_feature in X.columns:
        X[allowed_null_feature] = (
            X[allowed_null_feature].fillna(0.0)
        )

    X = X.astype(np.float32)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Features contain non-finite numeric values."
        )

    return X


def normalize_shap_values(
    values: Any,
    expected_rows: int,
    expected_features: int,
) -> np.ndarray:
    """Normalize SHAP output to a two-dimensional numeric array.

    SHAP versions may return an array, an Explanation object, or a list of
    arrays. This helper handles common binary-classification output formats.

    Args:
        values: Output returned by TreeExplainer.shap_values.
        expected_rows: Expected number of explained transactions.
        expected_features: Expected number of model features.

    Returns:
        A two-dimensional array of shape
        (expected_rows, expected_features).

    Raises:
        ValueError:
            If the output shape cannot be interpreted as expected.
    """
    if hasattr(values, "values"):
        values = values.values

    if isinstance(values, list):
        if len(values) == 1:
            values = values[0]
        elif len(values) == 2:
            # For two-class output, use the positive (fraud) class.
            values = values[1]

    values = np.asarray(values)

    if (
        values.ndim == 3
        and values.shape[0] == expected_rows
        and values.shape[1] == expected_features
        and values.shape[2] == 2
    ):
        values = values[:, :, 1]

    if values.shape != (
        expected_rows,
        expected_features,
    ):
        raise ValueError(
            "Unexpected SHAP output shape. "
            f"Expected {(expected_rows, expected_features)}, "
            f"received {values.shape}."
        )

    return values.astype(np.float64, copy=False)


def compute_shap_values(
    model: xgb.Booster,
    X_sample: pd.DataFrame,
) -> tuple[np.ndarray, shap.TreeExplainer]:
    """Compute tree-based SHAP values for the sampled validation rows.

    The explanations use the model's raw margin, which is the output before
    the sigmoid probability transformation. Tree-path-dependent attribution
    is used, so no separate background dataset is required.

    Args:
        model: Trained XGBoost Booster.
        X_sample: Prepared validation feature matrix.

    Returns:
        A tuple containing the normalized SHAP-value matrix and explainer.
    """
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    print("\nComputing SHAP values...")
    raw_values = explainer.shap_values(
        X_sample,
        check_additivity=False,
    )

    values = normalize_shap_values(
        raw_values,
        expected_rows=len(X_sample),
        expected_features=len(X_sample.columns),
    )

    print(f"SHAP matrix shape: {values.shape}")

    return values, explainer


def save_global_importance(
    shap_values: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Save a global feature-importance table based on SHAP values.

    Mean absolute SHAP measures the average magnitude of a feature's
    contribution to the model's raw output. Mean signed SHAP indicates
    whether its contributions tend to be positive or negative on this sample.

    Args:
        shap_values: SHAP matrix with rows as transactions and columns
            as features.
        feature_columns: Feature names in SHAP matrix order.

    Returns:
        A DataFrame sorted by descending mean absolute SHAP value.
    """
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_absolute_shap": np.mean(
                np.abs(shap_values),
                axis=0,
            ),
            "mean_signed_shap": np.mean(
                shap_values,
                axis=0,
            ),
        }
    )

    importance_df = importance_df.sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        np.arange(1, len(importance_df) + 1),
    )

    importance_df.to_csv(
        OUTPUT_DIRECTORY / "shap_global_importance.csv",
        index=False,
    )

    return importance_df


def save_shap_plots(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
) -> None:
    """Create and save global SHAP visualizations.

    The plots include a beeswarm summary, a global bar ranking, and a
    dependence plot for tx_count_10m.

    Args:
        shap_values: SHAP matrix for the sampled validation transactions.
        X_sample: Feature matrix used for SHAP computation.
    """
    print("\nSaving SHAP summary plots...")

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=list(X_sample.columns),
        max_display=len(X_sample.columns),
        show=False,
    )

    plt.title("Shepherd SHAP Summary — Validation Sample")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIRECTORY / "shap_summary_beeswarm.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=list(X_sample.columns),
        plot_type="bar",
        max_display=len(X_sample.columns),
        show=False,
    )

    plt.title("Shepherd Global SHAP Importance")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIRECTORY / "shap_global_bar.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    if "tx_count_10m" in X_sample.columns:
        shap.dependence_plot(
            "tx_count_10m",
            shap_values,
            X_sample,
            interaction_index=None,
            show=False,
        )

        plt.title(
            "SHAP Dependence — Transactions in Previous 10 Minutes"
        )
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIRECTORY / "shap_tx_count_10m_dependence.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close()


def save_local_explanations(
    sample_df: pd.DataFrame,
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    calibrated_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Save feature-level explanations for high-scoring validation examples.

    The ten sampled transactions with the highest sigmoid-calibrated
    probabilities are selected. Every model feature receives a row for each
    selected transaction, including its feature value and SHAP contribution.

    Positive contributions push the XGBoost raw margin toward fraud.
    Negative contributions push it away from fraud.

    Args:
        sample_df: Sampled validation rows with identifiers and labels.
        X_sample: Numeric model feature matrix in matching row order.
        shap_values: SHAP contribution matrix.
        calibrated_probabilities: Sigmoid-calibrated probabilities for
            the same sampled transactions.

    Returns:
        A long-form DataFrame containing transaction and feature explanations.
    """
    if len(calibrated_probabilities) != len(sample_df):
        raise ValueError(
            "Calibrated probabilities do not match the sampled rows."
        )

    example_count = min(
        LOCAL_EXAMPLE_COUNT,
        len(sample_df),
    )

    highest_risk_indices = np.argsort(
        calibrated_probabilities
    )[::-1][:example_count]

    rows: list[dict[str, Any]] = []

    for transaction_index in highest_risk_indices:
        transaction = sample_df.iloc[transaction_index]
        probability = float(
            calibrated_probabilities[transaction_index]
        )

        absolute_impacts = np.abs(
            shap_values[transaction_index]
        )

        feature_order = np.argsort(
            absolute_impacts
        )[::-1]

        for rank, feature_index in enumerate(
            feature_order,
            start=1,
        ):
            feature_name = X_sample.columns[feature_index]
            shap_contribution = float(
                shap_values[
                    transaction_index,
                    feature_index,
                ]
            )

            if shap_contribution > 0:
                direction = "pushes_toward_fraud"
            elif shap_contribution < 0:
                direction = "pushes_away_from_fraud"
            else:
                direction = "no_effect"

            rows.append(
                {
                    "transaction_id": str(
                        transaction[TRANSACTION_ID_COLUMN]
                    ),
                    "transaction_timestamp": str(
                        transaction[TIMESTAMP_COLUMN]
                    ),
                    "actual_is_fraud": int(
                        transaction[TARGET_COLUMN]
                    ),
                    "calibrated_fraud_probability": probability,
                    "operating_threshold": 0.23,
                    "predicted_fraud": int(
                        probability >= 0.23
                    ),
                    "feature_rank_by_absolute_impact": rank,
                    "feature": feature_name,
                    "feature_value": float(
                        X_sample.iloc[
                            transaction_index,
                            feature_index,
                        ]
                    ),
                    "shap_value_raw_margin": shap_contribution,
                    "absolute_shap_value": abs(
                        shap_contribution
                    ),
                    "contribution_direction": direction,
                }
            )

    local_df = pd.DataFrame(rows)

    local_df.to_csv(
        OUTPUT_DIRECTORY / "shap_local_explanations.csv",
        index=False,
    )

    return local_df


def main() -> None:
    """Run the SHAP explainability experiment and save all artifacts."""
    print("Shepherd SHAP explainability experiment")
    print("=" * 50)

    for artifact_path in (
        MODEL_PATH,
        CALIBRATORS_PATH,
        FEATURE_LIST_PATH,
    ):
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {artifact_path}"
            )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_columns = load_feature_columns()
    sample_df = load_validation_sample(feature_columns)
    X_sample = prepare_feature_matrix(
        sample_df,
        feature_columns,
    )

    print("\nLoading XGBoost model...")
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))

    print("Computing SHAP values...")
    shap_values, _ = compute_shap_values(
        model=model,
        X_sample=X_sample,
    )

    importance_df = save_global_importance(
        shap_values=shap_values,
        feature_columns=feature_columns,
    )

    save_shap_plots(
        shap_values=shap_values,
        X_sample=X_sample,
    )

    print("\nLoading sigmoid calibrator...")
    calibrators = joblib.load(CALIBRATORS_PATH)

    if CALIBRATION_METHOD not in calibrators:
        raise ValueError(
            f"Calibrator '{CALIBRATION_METHOD}' was not found."
        )

    dmatrix = xgb.DMatrix(
        X_sample,
        feature_names=feature_columns,
    )

    raw_margins = model.predict(
        dmatrix,
        output_margin=True,
    )

    calibrated_probabilities = calibrators[
        CALIBRATION_METHOD
    ].predict_proba(
        raw_margins.reshape(-1, 1)
    )[:, 1]

    local_df = save_local_explanations(
        sample_df=sample_df,
        X_sample=X_sample,
        shap_values=shap_values,
        calibrated_probabilities=calibrated_probabilities,
    )

    metadata = {
        "experiment_name": "shepherd_shap_explainability",
        "model_artifact": str(MODEL_PATH),
        "calibrator_artifact": str(CALIBRATORS_PATH),
        "feature_dataset": str(FEATURE_DATASET_PATH),
        "calibration_method": CALIBRATION_METHOD,
        "explained_output": (
            "XGBoost raw margin (log-odds), before sigmoid calibration"
        ),
        "shap_method": "TreeExplainer",
        "feature_perturbation": "tree_path_dependent",
        "validation_sample_size": int(len(X_sample)),
        "validation_sample_fraud_rate": float(
            sample_df[TARGET_COLUMN].mean()
        ),
        "local_examples_saved": int(
            local_df["transaction_id"].nunique()
        ),
        "local_example_selection": (
            "Highest sigmoid-calibrated fraud probabilities in the "
            "sampled validation dataset."
        ),
        "random_state": RANDOM_STATE,
        "global_importance_artifact": str(
            OUTPUT_DIRECTORY / "shap_global_importance.csv"
        ),
        "beeswarm_plot": str(
            OUTPUT_DIRECTORY / "shap_summary_beeswarm.png"
        ),
        "global_bar_plot": str(
            OUTPUT_DIRECTORY / "shap_global_bar.png"
        ),
        "dependence_plot": str(
            OUTPUT_DIRECTORY / "shap_tx_count_10m_dependence.png"
        ),
        "local_explanations_artifact": str(
            OUTPUT_DIRECTORY / "shap_local_explanations.csv"
        ),
        "notes": [
            "Only validation data was used for explanations.",
            "The reserved test period was not used.",
            "SHAP values explain model raw margins rather than calibrated "
            "probability differences.",
            "SHAP associations describe model behavior, not causal proof "
            "that a transaction is fraudulent.",
        ],
    }

    metadata_path = (
        OUTPUT_DIRECTORY / "shap_experiment_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nGlobal SHAP importance")
    print("-" * 65)
    print(
        importance_df[
            [
                "rank",
                "feature",
                "mean_absolute_shap",
                "mean_signed_shap",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_absolute_shap": "{:.5f}".format,
                "mean_signed_shap": "{:.5f}".format,
            },
        )
    )

    print("\nSaved explainability artifacts:")
    print(f"- {OUTPUT_DIRECTORY / 'shap_global_importance.csv'}")
    print(f"- {OUTPUT_DIRECTORY / 'shap_summary_beeswarm.png'}")
    print(f"- {OUTPUT_DIRECTORY / 'shap_global_bar.png'}")
    print(
        f"- {OUTPUT_DIRECTORY / 'shap_tx_count_10m_dependence.png'}"
    )
    print(f"- {OUTPUT_DIRECTORY / 'shap_local_explanations.csv'}")
    print(f"- {metadata_path}")

    print("\nSHAP explainability experiment completed.")

    del dmatrix
    del model
    gc.collect()


if __name__ == "__main__":
    main()