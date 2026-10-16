
"""Run an end-to-end smoke test of Shepherd's scoring component.

This module reads one transaction from the calibration experiment's validation
period, passes its feature values to FraudScorer, and prints the resulting
auditable decision as JSON.

The fraud target is not loaded or passed to the scorer. This is an inference
smoke test, not a performance evaluation.

Usage:
    python -m scoring.smoke_test
"""

import json

import pandas as pd

from scoring.predictor import (
    FEATURE_LIST_PATH,
    FraudScorer,
    MODEL_VERSION,
    PROJECT_ROOT,
)


FEATURE_DATASET_PATH = (
    PROJECT_ROOT
    / "artifacts/features/transaction_features.parquet"
)

VALIDATION_START_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85


def main() -> None:
    """Score one validation-period transaction and print the decision."""
    if not FEATURE_DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATASET_PATH}"
        )

    scorer = FraudScorer()

    required_columns = [
        "transaction_id",
        "transaction_timestamp",
        *scorer.feature_columns,
    ]

    # Do not load is_fraud: the scorer has no need for the target label.
    df = pd.read_parquet(
        FEATURE_DATASET_PATH,
        columns=required_columns,
    )

    if df.empty:
        raise ValueError("The feature dataset is empty.")

    if not pd.Index(df.columns).is_unique:
        raise ValueError(
            "The feature dataset contains duplicate column names."
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
        raise ValueError(
            "No rows were found in the validation partition."
        )

    # Take the first row in the validation period for a reproducible test.
    row = validation_df.iloc[0]

    features = {}

    for column in scorer.feature_columns:
        value = row[column]

        if pd.isna(value):
            features[column] = None
        else:
            features[column] = float(value)

    transaction_timestamp = pd.Timestamp(
        row["transaction_timestamp"]
    ).to_pydatetime()

    result = scorer.score_transaction(
        transaction_id=str(row["transaction_id"]),
        transaction_timestamp=transaction_timestamp,
        features=features,
    )

    print("Shepherd scoring smoke test")
    print("=" * 40)
    print(f"Model version: {MODEL_VERSION}")
    print(f"Feature schema: {FEATURE_LIST_PATH}")
    print("\nScoring result")
    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()