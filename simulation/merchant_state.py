"""
Represent the simulation state of a merchant.

This module defines the in-memory representation of a simulated merchant
before that merchant is persisted to PostgreSQL.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass
class MerchantState:
    """
    Represent a simulated merchant.

    Attributes
    ----------
    merchant_id:
        Unique identifier for the merchant.
    merchant_name:
        Human-readable simulated merchant name.
    merchant_category:
        Category assigned to the merchant.
    merchant_country:
        Two-letter country code in which the merchant operates.
    risk_score:
        Simulated baseline risk score for the merchant.
    """

    merchant_id: UUID
    merchant_name: str
    merchant_category: str
    merchant_country: str
    risk_score: float