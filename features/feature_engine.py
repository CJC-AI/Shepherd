"""
Compute leakage-safe fraud features from transaction history.

This module implements the first offline feature-engineering pass for
Shepherd. Transactions must be processed in chronological order. For every
transaction, only information from earlier transactions for the same customer
is used to construct historical features.

The implementation keeps customer history in memory while processing the
transaction stream. This avoids issuing one database history query per
transaction.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Iterable, Iterator
from uuid import UUID

from database.models import Transaction
from features.transaction_features import TransactionFeatures


@dataclass(frozen=True)
class MerchantReference:
    """
    Store merchant information required for feature computation.

    Attributes
    ----------
    merchant_category:
        Category assigned to the merchant.
    risk_score:
        Simulated baseline merchant risk score.
    """

    merchant_category: str
    risk_score: float


@dataclass(frozen=True)
class FeatureReferenceData:
    """
    Store small reference-data mappings required by the feature engine.

    Attributes
    ----------
    account_to_customer:
        Mapping from account UUID to customer UUID.
    customer_home_country:
        Mapping from customer UUID to home-country code.
    device_first_seen:
        Mapping from device UUID to first-seen timestamp.
    merchant_reference:
        Mapping from merchant UUID to merchant metadata.
    """

    account_to_customer: dict[UUID, UUID]
    customer_home_country: dict[UUID, str]
    device_first_seen: dict[UUID, datetime]
    merchant_reference: dict[UUID, MerchantReference]


@dataclass
class CustomerHistory:
    """
    Maintain historical state for one customer during feature computation.

    Attributes
    ----------
    transaction_count:
        Number of transactions processed before the current transaction.
    total_amount:
        Sum of transaction amounts processed before the current transaction.
    previous_timestamp:
        Timestamp of the customer's immediately preceding transaction.
    timestamps_10m:
        Timestamps for transactions in the 10-minute lookback window.
    timestamps_1h:
        Timestamps for transactions in the 1-hour lookback window.
    timestamps_24h:
        Timestamps for transactions in the 24-hour lookback window.
    merchant_categories_seen:
        Merchant categories encountered before the current transaction.
    """

    transaction_count: int = 0
    total_amount: float = 0.0
    previous_timestamp: datetime | None = None

    timestamps_10m: deque[datetime] = field(default_factory=deque)
    timestamps_1h: deque[datetime] = field(default_factory=deque)
    timestamps_24h: deque[datetime] = field(default_factory=deque)

    merchant_categories_seen: set[str] = field(
        default_factory=set
    )


class FeatureEngine:
    """
    Compute leakage-safe transaction features.

    The engine maintains historical state per customer while transactions are
    processed chronologically. The current transaction is not added to the
    customer's history until after its features have been calculated.

    Parameters
    ----------
    references:
        Reference-data mappings required to connect transactions to customers,
        devices, and merchants.
    """

    def __init__(
        self,
        references: FeatureReferenceData,
    ):
        """
        Initialize the feature engine.

        Parameters
        ----------
        references:
            Reference mappings used during feature computation.
        """

        self.references = references

        self.customer_history: dict[
            UUID,
            CustomerHistory,
        ] = defaultdict(CustomerHistory)

    def compute(
        self,
        transactions: Iterable[Transaction],
    ) -> Iterator[tuple[Transaction, TransactionFeatures]]:
        """
        Compute features for an ordered transaction stream.

        Transactions must be ordered chronologically. When two transactions
        have the same timestamp, their existing deterministic transaction
        ordering should be preserved by the caller.

        Parameters
        ----------
        transactions:
            Iterable of persisted Transaction ORM objects ordered by
            transaction timestamp.

        Yields
        ------
        TransactionFeatures
            Leakage-safe features for each transaction.

        Raises
        ------
        KeyError
            If a transaction references an account, device, or merchant that
            is missing from the reference data.
        """

        for transaction in transactions:
            customer_id = self._get_customer_id(
                transaction.account_id
            )

            merchant = self._get_merchant(
                transaction.merchant_id
            )

            device_first_seen = self._get_device_first_seen(
                transaction.device_id
            )

            history = self.customer_history[
                customer_id
            ]

            self._prune_history(
                history=history,
                current_timestamp=transaction.transaction_timestamp,
            )

            avg_amount_before = self._average_amount_before(
                history
            )

            amount = self._to_float(
                transaction.amount
            )

            amount_vs_avg = (
                amount / avg_amount_before
                if avg_amount_before > 0
                else 0.0
            )

            device_age_hours = max(
                0.0,
                (
                    transaction.transaction_timestamp
                    - device_first_seen
                ).total_seconds()
                / 3600.0,
            )

            previous_timestamp = (
                history.previous_timestamp
            )

            seconds_since_previous_tx = (
                (
                    transaction.transaction_timestamp
                    - previous_timestamp
                ).total_seconds()
                if previous_timestamp is not None
                else None
            )

            home_country = (
                self.references.customer_home_country[
                    customer_id
                ]
            )

            features = TransactionFeatures(
                transaction_id=transaction.transaction_id,
                transaction_timestamp=(
                    transaction.transaction_timestamp
                ),
                amount=amount,
                merchant_risk_score=merchant.risk_score,
                is_home_country=(
                    transaction.transaction_country
                    == home_country
                ),
                device_age_hours=device_age_hours,
                transactions_before=history.transaction_count,
                avg_amount_before=avg_amount_before,
                amount_vs_avg=amount_vs_avg,
                seconds_since_previous_tx=(
                    seconds_since_previous_tx
                ),
                tx_count_10m=len(
                    history.timestamps_10m
                ),
                tx_count_1h=len(
                    history.timestamps_1h
                ),
                tx_count_24h=len(
                    history.timestamps_24h
                ),
                merchant_category_seen_before=(
                    merchant.merchant_category
                    in history.merchant_categories_seen
                ),
            )

            yield transaction, features

            self._update_history(
                history=history,
                transaction=transaction,
                merchant_category=merchant.merchant_category,
            )

    def _get_customer_id(
        self,
        account_id: UUID,
    ) -> UUID:
        """
        Resolve an account identifier to its customer.

        Parameters
        ----------
        account_id:
            Account identifier from the current transaction.

        Returns
        -------
        UUID
            Owning customer identifier.

        Raises
        ------
        KeyError
            If the account is missing from the reference mapping.
        """

        try:
            return self.references.account_to_customer[
                account_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"Account {account_id} is missing from "
                "feature reference data."
            ) from exc

    def _get_merchant(
        self,
        merchant_id: UUID,
    ) -> MerchantReference:
        """
        Resolve a merchant identifier to its metadata.

        Parameters
        ----------
        merchant_id:
            Merchant identifier from the current transaction.

        Returns
        -------
        MerchantReference
            Merchant category and risk score.

        Raises
        ------
        KeyError
            If the merchant is missing from the reference mapping.
        """

        try:
            return self.references.merchant_reference[
                merchant_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"Merchant {merchant_id} is missing from "
                "feature reference data."
            ) from exc

    def _get_device_first_seen(
        self,
        device_id: UUID,
    ) -> datetime:
        """
        Resolve a device identifier to its first-seen timestamp.

        Parameters
        ----------
        device_id:
            Device identifier from the current transaction.

        Returns
        -------
        datetime
            Device first-seen timestamp.

        Raises
        ------
        KeyError
            If the device is missing from the reference mapping.
        """

        try:
            return self.references.device_first_seen[
                device_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"Device {device_id} is missing from "
                "feature reference data."
            ) from exc

    def _prune_history(
        self,
        history: CustomerHistory,
        current_timestamp: datetime,
    ) -> None:
        """
        Remove history outside the required lookback windows.

        Parameters
        ----------
        history:
            Customer history to update.
        current_timestamp:
            Timestamp of the current transaction.

        Returns
        -------
        None
        """

        cutoff_10m = (
            current_timestamp
            - timedelta(minutes=10)
        )

        cutoff_1h = (
            current_timestamp
            - timedelta(hours=1)
        )

        cutoff_24h = (
            current_timestamp
            - timedelta(hours=24)
        )

        while (
            history.timestamps_10m
            and history.timestamps_10m[0] < cutoff_10m
        ):
            history.timestamps_10m.popleft()

        while (
            history.timestamps_1h
            and history.timestamps_1h[0] < cutoff_1h
        ):
            history.timestamps_1h.popleft()

        while (
            history.timestamps_24h
            and history.timestamps_24h[0] < cutoff_24h
        ):
            history.timestamps_24h.popleft()

    def _average_amount_before(
        self,
        history: CustomerHistory,
    ) -> float:
        """
        Calculate the customer's historical average amount.

        Parameters
        ----------
        history:
            Customer history accumulated before the current transaction.

        Returns
        -------
        float
            Historical average amount, or zero when no prior transaction
            exists.
        """

        if history.transaction_count == 0:
            return 0.0

        return (
            history.total_amount
            / history.transaction_count
        )

    def _update_history(
        self,
        history: CustomerHistory,
        transaction: Transaction,
        merchant_category: str,
    ) -> None:
        """
        Add the current transaction to customer history.

        This method is intentionally called only after the current transaction's
        features have been computed, preventing target-row leakage.

        Parameters
        ----------
        history:
            Customer history to update.
        transaction:
            Current transaction.
        merchant_category:
            Category associated with the transaction's merchant.

        Returns
        -------
        None
        """

        timestamp = transaction.transaction_timestamp
        amount = self._to_float(transaction.amount)

        history.transaction_count += 1
        history.total_amount += amount
        history.previous_timestamp = timestamp

        history.timestamps_10m.append(
            timestamp
        )

        history.timestamps_1h.append(
            timestamp
        )

        history.timestamps_24h.append(
            timestamp
        )

        history.merchant_categories_seen.add(
            merchant_category
        )

    def _to_float(
        self,
        value: Decimal | float | int,
    ) -> float:
        """
        Convert a database numeric value to float.

        Parameters
        ----------
        value:
            Numeric database value.

        Returns
        -------
        float
            Floating-point representation.
        """

        return float(value)