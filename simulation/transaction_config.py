"""
Configuration for simulated transaction generation.

This module centralizes transaction-volume and fraud-scenario assumptions so
that the simulation can be tuned without modifying the generation logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransactionSimulationConfig:
    """
    Define transaction-generation parameters.

    Attributes
    ----------
    fraud_rate:
        Target approximate proportion of generated transactions that are
        fraud in the synthetic dataset.
    max_transactions_per_customer:
        Safety limit preventing a single customer from producing an
        unexpectedly large number of transactions in one generation pass.
    velocity_window_minutes:
        Time window used later for velocity-based fraud scenarios.
    new_device_window_days:
        Number of days during which a device is considered newly introduced.
    amount_anomaly_multiplier:
        Multiplier used when generating unusually large transaction amounts.
    """

    fraud_rate: float = 0.05
    max_transactions_per_customer: int = 1000
    velocity_window_minutes: int = 10
    new_device_window_days: int = 7
    amount_anomaly_multiplier: float = 4.0


DEFAULT_TRANSACTION_CONFIG = TransactionSimulationConfig()