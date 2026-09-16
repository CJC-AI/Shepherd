"""
Generate simulated merchants for the Shepherd simulation.

This module produces deterministic MerchantState objects using the merchant
categories and country codes defined by the simulation constants.
"""

import random
import uuid

from simulation.constants import COUNTRIES, MERCHANT_CATEGORIES
from simulation.merchant_state import MerchantState


MERCHANT_NAMESPACE = uuid.UUID(
    "b6c6f6a7-8f5d-46b7-9e84-5f8d5a2b1c03"
)


CATEGORY_BASE_RISK = {
    "groceries": 0.10,
    "transport": 0.15,
    "pharmacy": 0.12,
    "utilities": 0.08,
    "restaurants": 0.18,
    "subscriptions": 0.12,
    "electronics": 0.25,
    "hotels": 0.22,
    "travel": 0.30,
    "luxury": 0.28,
    "gaming": 0.32,
    "digital_goods": 0.35,
    "e_commerce": 0.27,
    "food_delivery": 0.20,
}


class MerchantGenerator:
    """
    Generate a reproducible merchant population.

    Parameters
    ----------
    seed:
        Optional random seed controlling merchant category and country
        selection and small variations in risk score.
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize the merchant generator.

        Parameters
        ----------
        seed:
            Optional random seed used for reproducible merchant attributes.
        """
        self.random = random.Random(seed)

    def generate(
        self,
        count: int = 500,
    ) -> list[MerchantState]:
        """
        Generate a collection of simulated merchants.

        Parameters
        ----------
        count:
            Number of merchants to generate.

        Returns
        -------
        list[MerchantState]
            Generated merchant states.

        Raises
        ------
        ValueError
            If count is less than or equal to zero.
            If a configured merchant category does not have a risk baseline.
        """

        if count <= 0:
            raise ValueError("count must be greater than zero")

        missing_categories = [
            category
            for category in MERCHANT_CATEGORIES
            if category not in CATEGORY_BASE_RISK
        ]

        if missing_categories:
            raise ValueError(
                "Missing risk scores for merchant categories: "
                f"{missing_categories}"
            )

        merchants: list[MerchantState] = []

        for index in range(count):
            merchant_id = uuid.uuid5(
                MERCHANT_NAMESPACE,
                f"merchant-{index}",
            )

            category = self.random.choice(
                MERCHANT_CATEGORIES
            )

            country = self.random.choice(
                COUNTRIES
            )

            risk_score = self._generate_risk_score(
                category
            )

            merchant_name = self._generate_name(
                index=index,
                category=category,
                country=country,
            )

            merchants.append(
                MerchantState(
                    merchant_id=merchant_id,
                    merchant_name=merchant_name,
                    merchant_category=category,
                    merchant_country=country,
                    risk_score=risk_score,
                )
            )

        return merchants

    def _generate_risk_score(
        self,
        category: str,
    ) -> float:
        """
        Generate a simulated merchant risk score.

        The category-specific baseline represents a simulation assumption,
        not an empirical estimate of real-world merchant fraud risk.

        Parameters
        ----------
        category:
            Merchant category used to select the risk baseline.

        Returns
        -------
        float
            Risk score constrained to the interval [0.0, 1.0].
        """

        baseline = CATEGORY_BASE_RISK[category]

        variation = self.random.uniform(
            -0.05,
            0.05,
        )

        return round(
            min(
                1.0,
                max(
                    0.0,
                    baseline + variation,
                ),
            ),
            4,
        )

    def _generate_name(
        self,
        index: int,
        category: str,
        country: str,
    ) -> str:
        """
        Generate a deterministic human-readable merchant name.

        Parameters
        ----------
        index:
            Position of the merchant in the generated population.
        category:
            Merchant category.
        country:
            Merchant country code.

        Returns
        -------
        str
            Simulated merchant name.
        """

        category_name = category.replace(
            "_",
            " ",
        ).title()

        return (
            f"{category_name} Merchant "
            f"{country}-{index + 1:04d}"
        )