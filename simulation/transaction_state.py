"""
Represent the simulation state of a bank transaction.

This module defines the in-memory transaction representation used by the
Shepherd simulation before transactions are persisted to PostgreSQL.

The simulation state contains transaction-level information, including
transaction_country, which is useful for generating and validating synthetic
geographic behavior. The persistence layer may choose which fields are stored
in the database.
"""

from dataclasses import dataclass
from datetime import datetime
from ipaddress import IPv4Address
from uuid import UUID


@dataclass
class TransactionState:
    """
    Represent one simulated bank transaction.

    Attributes
    ----------
    transaction_id:
        Unique identifier for the transaction.
    account_id:
        Account from which the transaction originates.
    merchant_id:
        Merchant receiving the transaction.
    device_id:
        Device used to initiate the transaction.
    amount:
        Transaction amount in the account's transaction currency.
    currency:
        Three-letter ISO-style currency code used for the transaction.
    transaction_timestamp:
        Simulation timestamp at which the transaction occurs.
    transaction_country:
        Country in which the transaction is simulated to occur. This is
        simulation metadata used to model geographic behavior and may later be
        persisted or replaced by a derived location representation.
    ip_address:
        IPv4 address associated with the transaction.
    is_fraud:
        Optional ground-truth fraud label. This is None when the transaction
        has not yet been labelled.
    """

    transaction_id: UUID
    account_id: UUID
    merchant_id: UUID
    device_id: UUID
    amount: float
    currency: str
    transaction_timestamp: datetime
    transaction_country: str
    ip_address: IPv4Address
    is_fraud: bool | None = None