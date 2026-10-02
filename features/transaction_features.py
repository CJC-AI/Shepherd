"""
Define the feature representation used by Shepherd fraud detection.

This module contains the leakage-safe feature contract for a transaction.
Features describe information that would have been available at the time
the transaction was evaluated.

The feature representation deliberately excludes the ground-truth fraud
label, future transactions, and internal synthetic fraud-scenario metadata.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class TransactionFeatures:
    """
    Represent the fraud features for one transaction.

    Attributes
    ----------
    transaction_id:
        Identifier of the transaction these features describe.
    transaction_timestamp:
        Timestamp at which the transaction occurred.

    amount:
        Current transaction amount.
    merchant_risk_score:
        Baseline risk score assigned to the merchant.
    is_home_country:
        Whether the transaction occurred in the customer's home country.
    device_age_hours:
        Number of hours between device first_seen and the transaction.

    transactions_before:
        Number of customer transactions that occurred before the current
        transaction.
    avg_amount_before:
        Customer's average transaction amount before the current transaction.
    amount_vs_avg:
        Current amount divided by the customer's historical average amount.

    seconds_since_previous_tx:
        Seconds since the customer's immediately preceding transaction.

    tx_count_10m:
        Number of previous customer transactions in the preceding 10 minutes.
    tx_count_1h:
        Number of previous customer transactions in the preceding hour.
    tx_count_24h:
        Number of previous customer transactions in the preceding 24 hours.

    merchant_category_seen_before:
        Whether the customer has previously used the current merchant
        category before this transaction.
    """

    transaction_id: UUID
    transaction_timestamp: datetime

    amount: float
    merchant_risk_score: float
    is_home_country: bool
    device_age_hours: float

    transactions_before: int
    avg_amount_before: float
    amount_vs_avg: float

    seconds_since_previous_tx: float | None

    tx_count_10m: int
    tx_count_1h: int
    tx_count_24h: int

    merchant_category_seen_before: bool