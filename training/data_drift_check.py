"""
Compare feature distributions across chronological dataset splits.

This module performs a lightweight descriptive comparison of model features
between the training, validation, and test periods. It does not train a model
and does not attempt to declare statistical significance.
"""

from pathlib import Path

import pandas as pd

from training.dataset_loader import MODEL_FEATURES
from training.training_config import DEFAULT_TRAINING_CONFIG


def compare_split_means(
    path: str = DEFAULT_TRAINING_CONFIG.artifact_path,
) -> None:
    """
    Compare mean feature values across chronological splits.

    Parameters
    ----------
    path:
        Path to the Parquet feature dataset.

    Returns
    -------
    None
    """

    dataframe = pd.read_parquet(path)

    dataframe = dataframe.sort_values(
        by=[
            "transaction_timestamp",
            "transaction_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    row_count = len(dataframe)

    train_end = int(
        row_count
        * DEFAULT_TRAINING_CONFIG.train_fraction
    )

    validation_end = train_end + int(
        row_count
        * DEFAULT_TRAINING_CONFIG.validation_fraction
    )

    splits = {
        "train": dataframe.iloc[:train_end],
        "validation": dataframe.iloc[
            train_end:validation_end
        ],
        "test": dataframe.iloc[
            validation_end:
        ],
    }

    summary_rows = []

    for feature in MODEL_FEATURES:
        row = {"feature": feature}

        for split_name, split in splits.items():
            row[split_name] = split[feature].mean()

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    print(summary.to_string(index=False))


def main() -> None:
    """
    Run the chronological feature-distribution comparison.

    Returns
    -------
    None
    """

    compare_split_means()


if __name__ == "__main__":
    main()