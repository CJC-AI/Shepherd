"""
Generate simulated bank transactions for Shepherd customers.

This module generates transaction-level simulation state from already-created
customers, accounts, devices, and merchants.

The first version focuses on generating realistic baseline transactions.
Fraud scenarios are intentionally handled separately so that normal
transaction behavior can be validated before synthetic fraud is introduced.
"""

import random
import uuid
import hashlib
from datetime import datetime, timedelta
from ipaddress import IPv4Address
from typing import Sequence

from simulation.account_state import AccountState
from simulation.customer_state import CustomerState
from simulation.device_state import DeviceState
from simulation.merchant_state import MerchantState
from simulation.transaction_config import (
    DEFAULT_TRANSACTION_CONFIG,
    TransactionSimulationConfig,
)
from simulation.transaction_state import TransactionState


class TransactionGenerator:
    """
    Generate baseline transactions for simulated customers.

    Parameters
    ----------
    config:
        Transaction-generation configuration.
    seed:
        Optional random seed for reproducible transaction generation.
    """

    def __init__(
        self,
        config: TransactionSimulationConfig = DEFAULT_TRANSACTION_CONFIG,
        seed: int | None = None,
    ):
        """
        Initialize the transaction generator.

        Parameters
        ----------
        config:
            Configuration controlling transaction generation.
        seed:
            Optional random seed used for reproducibility.
        """

        self.config = config
        self.random = random.Random(seed)

    def generate_for_customer(
        self,
        customer: CustomerState,
        accounts: Sequence[AccountState],
        devices: Sequence[DeviceState],
        merchants: Sequence[MerchantState],
    ) -> list[TransactionState]:
        """
        Generate baseline transactions for one customer.

        Transactions are generated across the customer's active lifecycle.
        A device can only be selected after its first_seen timestamp, and
        merchants are selected with preference toward the customer's preferred
        merchant categories.

        Parameters
        ----------
        customer:
            Customer whose activity is being simulated.
        accounts:
            Accounts belonging to the customer.
        devices:
            Devices belonging to the customer.
        merchants:
            Merchant catalog available to the customer.

        Returns
        -------
        list[TransactionState]
            Generated baseline transactions.

        Raises
        ------
        ValueError
            If no accounts, devices, or merchants are available.
        """

        if not accounts:
            raise ValueError(
                f"Customer {customer.customer_id} has no accounts."
            )

        if not devices:
            raise ValueError(
                f"Customer {customer.customer_id} has no devices."
            )

        if not merchants:
            raise ValueError("At least one merchant is required.")

        active_days = (
            customer.customer_end_time
            - customer.customer_start_time
        ).days

        if active_days <= 0:
            active_days = 1

        transaction_count = min(
            customer.transactions_per_day * active_days,
            self.config.max_transactions_per_customer,
        )

        transactions: list[TransactionState] = []

        for transaction_index in range(transaction_count):
            transaction_timestamp = self._generate_timestamp(
                customer=customer
            )

            available_devices = [
                device
                for device in devices
                if device.first_seen <= transaction_timestamp
            ]

            if not available_devices:
                available_devices = [devices[0]]

            device = self.random.choice(
                available_devices
            )

            account = self.random.choice(
                list(accounts)
            )

            merchant = self._choose_merchant(
                customer=customer,
                merchants=merchants,
            )

            amount = self._generate_amount(
                customer=customer
            )

            transaction_country = self._choose_country(
                customer=customer
            )

            ip_address = self._generate_ip_address(
                customer=customer,
                transaction_country=transaction_country,
            )

            transactions.append(
                TransactionState(
                    transaction_id=uuid.uuid5(
                        customer.customer_id,
                        f"transaction-{transaction_index}",
                    ),
                    account_id=account.account_id,
                    merchant_id=merchant.merchant_id,
                    device_id=device.device_id,
                    amount=amount,
                    currency=account.currency,
                    transaction_timestamp=transaction_timestamp,
                    transaction_country=transaction_country,
                    ip_address=ip_address,
                    is_fraud=None,
                )
            )

        transactions.sort(
            key=lambda transaction: transaction.transaction_timestamp
        )

        return transactions

    def _generate_timestamp(
        self,
        customer: CustomerState,
    ) -> datetime:
        """
        Generate a random transaction timestamp inside the customer lifecycle.

        Parameters
        ----------
        customer:
            Customer whose lifecycle defines the valid timestamp range.

        Returns
        -------
        datetime
            Timezone-aware transaction timestamp.
        """

        total_seconds = (
            customer.customer_end_time
            - customer.customer_start_time
        ).total_seconds()

        if total_seconds <= 0:
            return customer.customer_start_time

        offset_seconds = self.random.uniform(
            0,
            total_seconds,
        )

        return customer.customer_start_time + timedelta(
            seconds=offset_seconds
        )

    def _generate_amount(
        self,
        customer: CustomerState,
    ) -> float:
        """
        Generate a transaction amount around the customer's spending baseline.

        Parameters
        ----------
        customer:
            Customer whose spending behavior controls the amount distribution.

        Returns
        -------
        float
            Positive transaction amount rounded to two decimals.
        """

        amount = self.random.gauss(
            customer.average_spend,
            customer.spend_std,
        )

        return round(
            max(0.50, amount),
            2,
        )

    def _choose_merchant(
        self,
        customer: CustomerState,
        merchants: Sequence[MerchantState],
    ) -> MerchantState:
        """
        Select a merchant with preference toward customer-familiar categories.

        Parameters
        ----------
        customer:
            Customer whose preferences influence merchant selection.
        merchants:
            Merchant catalog.

        Returns
        -------
        MerchantState
            Selected merchant.
        """

        preferred_merchants = [
            merchant
            for merchant in merchants
            if merchant.merchant_category
            in customer.preferred_merchant_categories
        ]

        if preferred_merchants and self.random.random() < 0.80:
            return self.random.choice(
                preferred_merchants
            )

        return self.random.choice(
            list(merchants)
        )

    def _choose_country(
        self,
        customer: CustomerState,
    ) -> str:
        """
        Choose a transaction country using the customer's travel behavior.

        Parameters
        ----------
        customer:
            Customer whose usual countries and international probability
            determine the transaction country.

        Returns
        -------
        str
            Two-letter country code.
        """

        international = (
            self.random.random()
            < customer.archetype.international_probability
        )

        if not international:
            return customer.home_country

        international_countries = [
            country
            for country in customer.usual_countries
            if country != customer.home_country
        ]

        if not international_countries:
            return customer.home_country

        return self.random.choice(
            international_countries
        )

    def _generate_ip_address(
        self,
        customer: CustomerState,
        transaction_country: str,
    ) -> IPv4Address:
        """
        Generate a reproducible synthetic IPv4 address.

        The address is derived from the customer identifier, transaction country,
        and generator randomness using SHA-256. Unlike Python's built-in hash(),
        SHA-256 produces stable output across separate Python processes.

        Parameters
        ----------
        customer:
            Customer associated with the transaction.
        transaction_country:
            Country where the transaction is simulated to occur.

        Returns
        -------
        IPv4Address
            Reproducible synthetic IPv4 address.
        """

        random_component = self.random.randint(
            0,
            2**32 - 1,
        )

        seed_value = (
            f"{customer.customer_id}-"
            f"{transaction_country}-"
            f"{random_component}"
        )

        digest = hashlib.sha256(
            seed_value.encode("utf-8")
        ).digest()

        numeric_value = int.from_bytes(
            digest[:4],
            byteorder="big",
        )

        return IPv4Address(numeric_value)