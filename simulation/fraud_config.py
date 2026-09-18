"""
Configuration for synthetic fraud scenario generation.

This module centralizes the assumptions controlling synthetic fraud injection.
The values are simulation parameters and are not intended to represent real
fraud prevalence or observed fraud-pattern distributions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FraudScenarioConfig:
    """
    Define synthetic fraud-generation parameters.

    Attributes
    ----------
    fraud_rate:
        Approximate proportion of transactions that should receive a fraud
        label.
    new_device_weight:
        Relative weight of the new-device scenario.
    geographic_anomaly_weight:
        Relative weight of the geographic-anomaly scenario.
    amount_anomaly_weight:
        Relative weight of the amount-anomaly scenario.
    velocity_weight:
        Relative weight of the velocity scenario.
    new_device_window_days:
        Maximum number of days after a device is first seen during which the
        device can be used for a synthetic new-device fraud scenario.
    amount_anomaly_multiplier:
        Multiplier applied to a baseline transaction amount when creating an
        amount-anomaly scenario.
    velocity_window_minutes:
        Maximum temporal separation used to create a synthetic velocity burst.
    """

    fraud_rate: float = 0.05

    new_device_weight: float = 0.30
    geographic_anomaly_weight: float = 0.25
    amount_anomaly_weight: float = 0.25
    velocity_weight: float = 0.20

    new_device_window_days: int = 7
    amount_anomaly_multiplier: float = 4.0
    velocity_window_minutes: int = 10


DEFAULT_FRAUD_SCENARIO_CONFIG = FraudScenarioConfig()