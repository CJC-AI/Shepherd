"""
Validate the chronological Shepherd training dataset split.

This test loads the complete feature dataset, creates the chronological
train/validation/test partitions, and verifies their sizes, ordering, feature
dimensions, target balance, and temporal boundaries.
"""

from training.dataset_loader import (
    MODEL_FEATURES,
    load_feature_dataset,
    split_dataset,
)


def main() -> None:
    """
    Validate the chronological dataset split.

    Returns
    -------
    None
    """

    dataframe = load_feature_dataset()

    split = split_dataset(
        dataframe
    )

    print(
        "total rows:",
        len(dataframe),
    )

    print(
        "train rows:",
        len(split.X_train),
    )

    print(
        "validation rows:",
        len(split.X_validation),
    )

    print(
        "test rows:",
        len(split.X_test),
    )

    print(
        "feature count:",
        split.X_train.shape[1],
    )

    print(
        "train fraud rate:",
        split.y_train.mean(),
    )

    print(
        "validation fraud rate:",
        split.y_validation.mean(),
    )

    print(
        "test fraud rate:",
        split.y_test.mean(),
    )

    print(
        "train end:",
        split.train_end_timestamp,
    )

    print(
        "validation end:",
        split.validation_end_timestamp,
    )

    print(
        "test starts after validation:",
        split.validation_end_timestamp
        <= dataframe[
            "transaction_timestamp"
        ].iloc[
            len(split.X_train)
            + len(split.X_validation)
        ],
    )

    assert (
        split.X_train.shape[1]
        == len(MODEL_FEATURES)
    )

    assert (
        len(split.X_train)
        + len(split.X_validation)
        + len(split.X_test)
        == len(dataframe)
    )

    assert (
        split.train_end_timestamp
        <= split.validation_end_timestamp
    )

    print(
        "chronological dataset split: PASS"
    )


if __name__ == "__main__":
    main()