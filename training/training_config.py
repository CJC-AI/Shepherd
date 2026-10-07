"""
Define configuration for Shepherd model training.

This module centralizes reproducibility, temporal split, and model-training
parameters so that training experiments can be repeated consistently without
hard-coding values throughout the training pipeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """
    Define model-training configuration.

    Attributes
    ----------
    train_fraction:
        Fraction of chronologically ordered transactions assigned to training.
    validation_fraction:
        Fraction assigned to validation after the training period.
    test_fraction:
        Fraction assigned to the final chronological test period.
    random_state:
        Random seed used by model-training components that support stochastic
        behavior.
    target_column:
        Name of the supervised fraud label column.
    artifact_path:
        Location of the offline feature dataset.
    """

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 42
    target_column: str = "is_fraud"
    artifact_path: str = (
        "artifacts/features/transaction_features.parquet"
    )

    def __post_init__(self) -> None:
        """
        Validate the training configuration.

        Raises
        ------
        ValueError
            If split fractions are invalid or do not sum to one.
        """

        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )

        if any(
            fraction <= 0 or fraction >= 1
            for fraction in fractions
        ):
            raise ValueError(
                "All split fractions must be greater than 0 "
                "and less than 1."
            )

        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError(
                "Train, validation, and test fractions must sum to 1.0."
            )


DEFAULT_TRAINING_CONFIG = TrainingConfig()