"""
Load and split the Shepherd offline feature dataset.

This module loads the validated Parquet feature dataset, preserves its
chronological ordering, converts model features to memory-efficient numeric
types, and creates deterministic train, validation, and test partitions.

The target label is kept separate from the model feature matrix.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from training.training_config import DEFAULT_TRAINING_CONFIG


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


@dataclass
class DatasetSplit:
    """
    Store the chronological training dataset partitions.

    Attributes
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Training fraud labels.
    X_validation:
        Validation feature matrix.
    y_validation:
        Validation fraud labels.
    X_test:
        Test feature matrix.
    y_test:
        Test fraud labels.
    train_end_timestamp:
        Final transaction timestamp included in training.
    validation_end_timestamp:
        Final transaction timestamp included in validation.
    """

    X_train: np.ndarray
    y_train: np.ndarray

    X_validation: np.ndarray
    y_validation: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray

    train_end_timestamp: pd.Timestamp
    validation_end_timestamp: pd.Timestamp


def load_feature_dataset(
    path: str = DEFAULT_TRAINING_CONFIG.artifact_path,
) -> pd.DataFrame:
    """
    Load the validated feature dataset from Parquet.

    Parameters
    ----------
    path:
        Path to the Parquet feature dataset.

    Returns
    -------
    pandas.DataFrame
        Chronologically ordered feature dataset.
    """

    dataframe = pd.read_parquet(path)

    dataframe = dataframe.sort_values(
        by=[
            "transaction_timestamp",
            "transaction_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return dataframe


def split_dataset(
    dataframe: pd.DataFrame,
) -> DatasetSplit:
    """
    Split the feature dataset chronologically.

    The split is performed by row position after chronological ordering.
    No shuffling is performed.

    Parameters
    ----------
    dataframe:
        Validated feature dataset.

    Returns
    -------
    DatasetSplit
        Chronological train, validation, and test partitions.
    """

    config = DEFAULT_TRAINING_CONFIG

    row_count = len(dataframe)

    train_end = int(
        row_count * config.train_fraction
    )

    validation_end = train_end + int(
        row_count * config.validation_fraction
    )

    train = dataframe.iloc[
        :train_end
    ]

    validation = dataframe.iloc[
        train_end:validation_end
    ]

    test = dataframe.iloc[
        validation_end:
    ]

    X_train = _prepare_features(train)
    X_validation = _prepare_features(validation)
    X_test = _prepare_features(test)

    y_train = train[
        config.target_column
    ].to_numpy(dtype=np.int8)

    y_validation = validation[
        config.target_column
    ].to_numpy(dtype=np.int8)

    y_test = test[
        config.target_column
    ].to_numpy(dtype=np.int8)

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        train_end_timestamp=(
            train["transaction_timestamp"].iloc[-1]
        ),
        validation_end_timestamp=(
            validation[
                "transaction_timestamp"
            ].iloc[-1]
        ),
    )


def _prepare_features(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """
    Convert model features to a memory-efficient NumPy matrix.

    Parameters
    ----------
    dataframe:
        Dataset partition containing the model features.

    Returns
    -------
    numpy.ndarray
        Float32 feature matrix.
    """

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )

    features = dataframe[
        MODEL_FEATURES
    ].copy()

    features["is_home_country"] = (
        features["is_home_country"]
        .astype(np.float32)
    )

    features[
        "merchant_category_seen_before"
    ] = (
        features[
            "merchant_category_seen_before"
        ].astype(np.float32)
    )

    features = features.fillna(0.0)

    return features.to_numpy(
        dtype=np.float32
    )