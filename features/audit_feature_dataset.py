"""
Audit the Shepherd offline feature dataset.

This module performs structural and statistical sanity checks on the
Parquet feature dataset before it is passed to model training.

The audit checks row counts, duplicate transaction identifiers, null values,
non-finite numeric values, feature ranges, target balance, and chronological
ordering.
"""

from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


FEATURE_PATH = Path(
    "artifacts/features/transaction_features.parquet"
)


NUMERIC_COLUMNS = [
    "amount",
    "merchant_risk_score",
    "device_age_hours",
    "transactions_before",
    "avg_amount_before",
    "amount_vs_avg",
    "seconds_since_previous_tx",
    "tx_count_10m",
    "tx_count_1h",
    "tx_count_24h",
]


def audit_feature_dataset(
    path: Path = FEATURE_PATH,
) -> None:
    """
    Audit the persisted feature dataset.

    Parameters
    ----------
    path:
        Path to the Parquet feature dataset.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the feature dataset does not exist.
    AssertionError
        If any structural or numeric integrity check fails.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}"
        )

    table = pq.read_table(path)

    row_count = table.num_rows

    transaction_ids = table[
        "transaction_id"
    ]

    unique_transaction_ids = pc.count_distinct(
        transaction_ids
    ).as_py()

    assert unique_transaction_ids == row_count, (
        "Duplicate transaction IDs detected."
    )

    is_fraud = table["is_fraud"]

    fraud_count = pc.sum(
        is_fraud.cast("int64")
    ).as_py()

    non_fraud_count = (
        row_count - fraud_count
    )

    null_counts: dict[str, int] = {}

    for column_name in table.column_names:
        null_counts[column_name] = table[
            column_name
        ].null_count

    allowed_nullable_columns = {
        "seconds_since_previous_tx",
    }

    unexpected_nulls = {
        column_name: count
        for column_name, count in null_counts.items()
        if count > 0
        and column_name not in allowed_nullable_columns
    }

    assert not unexpected_nulls, (
        "Unexpected null values detected: "
        f"{unexpected_nulls}"
    )

    assert null_counts["seconds_since_previous_tx"] > 0, (
        "Expected first transactions to have"
        "NULL seconds_since_previous_tx values."
    )

    for column_name in NUMERIC_COLUMNS:
        column = table[column_name]

        non_null_column = column.drop_null()

        values = non_null_column.to_numpy(
            zero_copy_only=False
        )

        assert np.all(
            np.isfinite(values)
        ), (
            f"Non-finite values detected in {column_name}."
        )

    device_age = (
        table["device_age_hours"]
        .to_numpy(
            zero_copy_only=False
        )
    )

    assert np.all(
        device_age >= 0
    ), "Negative device ages detected."

    transaction_counts = [
        "transactions_before",
        "tx_count_10m",
        "tx_count_1h",
        "tx_count_24h",
    ]

    for column_name in transaction_counts:
        values = (
            table[column_name]
            .to_numpy(
                zero_copy_only=False
            )
        )

        assert np.all(values >= 0), (
            f"Negative values detected in {column_name}."
        )

    merchant_risk = (
        table["merchant_risk_score"]
        .to_numpy(
            zero_copy_only=False
        )
    )

    assert np.all(
        (merchant_risk >= 0)
        & (merchant_risk <= 1)
    ), (
        "Merchant risk score outside [0, 1]."
    )

    amount = (
        table["amount"]
        .to_numpy(
            zero_copy_only=False
        )
    )

    assert np.all(
        amount > 0
    ), "Non-positive transaction amounts detected."

    target_values = set(
        is_fraud.to_pylist()
    )

    assert target_values.issubset(
        {True, False}
    ), (
        f"Unexpected target values: {target_values}"
    )

    timestamps = (
        table["transaction_timestamp"]
        .to_pylist()
    )

    assert timestamps == sorted(
        timestamps
    ), (
        "Feature dataset is not globally chronological."
    )

    print(
        f"rows: {row_count:,}"
    )

    print(
        f"unique transaction IDs: "
        f"{unique_transaction_ids:,}"
    )

    print(
        f"fraud: {fraud_count:,}"
    )

    print(
        f"non-fraud: {non_fraud_count:,}"
    )

    print(
        f"fraud rate: "
        f"{fraud_count / row_count:.4%}"
    )

    print(
        "unexpected null values: 0"
    )

    print(
        "seconds_since_previous_tx nulls: "
        f"{null_counts['seconds_since_previous_tx']:,}"
    )

    print(
        "non-finite numeric values: 0"
    )

    print(
        "feature range checks: PASS"
    )

    print(
        "chronological ordering: PASS"
    )

    print(
        "feature dataset audit: PASS"
    )


def main() -> None:
    """
    Run the feature-dataset audit.

    Returns
    -------
    None
    """

    audit_feature_dataset()


if __name__ == "__main__":
    main()