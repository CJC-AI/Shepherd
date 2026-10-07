"""
Define the baseline XGBoost configuration for Shepherd.

This module centralizes model hyperparameters so that the baseline experiment
can be reproduced exactly and later compared with tuned model variants.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class XGBoostConfig:
    """
    Define baseline XGBoost hyperparameters.

    Attributes
    ----------
    n_estimators:
        Maximum number of boosting rounds. Early stopping may terminate
        training before this limit.
    learning_rate:
        Step size applied during boosting.
    max_depth:
        Maximum depth of each decision tree.
    min_child_weight:
        Minimum sum of instance weight required in a child node.
    subsample:
        Fraction of training observations sampled for each boosting round.
    colsample_bytree:
        Fraction of features sampled for each tree.
    reg_lambda:
        L2 regularization strength.
    reg_alpha:
        L1 regularization strength.
    random_state:
        Seed used by XGBoost.
    early_stopping_rounds:
        Number of validation rounds without improvement before stopping.
    """

    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: int = 5
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    random_state: int = 42
    early_stopping_rounds: int = 50


DEFAULT_XGBOOST_CONFIG = XGBoostConfig()