"""
Inject synthetic fraud scenarios into baseline transactions.

This module separates fraud-scenario generation from normal transaction
generation. Baseline transactions are first created from ordinary customer
behavior, after which this engine modifies a small subset to exhibit
observable characteristics associated with synthetic fraud.

Scenario identifiers are intentionally kept outside TransactionState so that
the generated fraud label does not directly reveal which scenario caused it.
"""

import hashlib
import random
import uuid
from dataclasses import dataclass, replace
from datetime import timedelta
from ipaddress import IPv4Address
from typing import Sequence

from simulation.constants import COUNTRIES
from simulation.customer_state import CustomerState
from simulation.device_state import DeviceState
from simulation.fraud_config import (
    DEFAULT_FRAUD_SCENARIO_CONFIG,
    FraudScenarioConfig,
)
from simulation.transaction_state import TransactionState


@dataclass
class FraudInjectionResult:
    """
    Store the result of synthetic fraud injection.

    Attributes
    ----------
    transactions:
        Transactions after fraud scenarios have been applied.
    scenario_counts:
        Number of transactions assigned to each synthetic fraud scenario.
        This metadata is for simulation diagnostics only and is not attached
        to individual TransactionState objects.
    """

    transactions: list[TransactionState]
    scenario_counts: dict[str, int]


class FraudScenarioEngine:
    """
    Inject synthetic fraud scenarios into a customer's transactions.

    Parameters
    ----------
    config:
        Configuration controlling fraud prevalence and scenario weights.
    seed:
        Optional random seed used for reproducible scenario selection.
    """

    def __init__(
        self,
        config: FraudScenarioConfig = DEFAULT_FRAUD_SCENARIO_CONFIG,
        seed: int | None = None,
    ):
        """
        Initialize the fraud scenario engine.

        Parameters
        ----------
        config:
            Synthetic fraud-generation configuration.
        seed:
            Optional random seed controlling reproducible fraud injection.
        """

        self.config = config
        self.random = random.Random(seed)

    def apply_for_customer(
        self,
        customer: CustomerState,
        transactions: Sequence[TransactionState],
        devices: Sequence[DeviceState],
    ) -> FraudInjectionResult:
        """
        Inject synthetic fraud into one customer's transaction set.

        The configured fraud rate is applied approximately to the supplied
        transaction batch. Normal transactions receive an explicit
        ``is_fraud=False`` label, while transactions selected for synthetic
        fraud receive ``is_fraud=True``.

        Parameters
        ----------
        customer:
            Customer whose transactions are being processed.
        transactions:
            Baseline transactions generated for the customer.
        devices:
            Devices belonging to the customer.

        Returns
        -------
        FraudInjectionResult
            Modified transactions and diagnostic scenario counts.

        Raises
        ------
        ValueError
            If the fraud rate is outside [0.0, 1.0].
            If a configured scenario weight is negative.
        """

        self._validate_configuration()

        working_transactions = [
            replace(
                transaction,
                is_fraud=False,
            )
            for transaction in transactions
        ]

        if not working_transactions:
            return FraudInjectionResult(
                transactions=[],
                scenario_counts=self._empty_scenario_counts(),
            )

        fraud_count = int(
            round(
                len(working_transactions)
                * self.config.fraud_rate
            )
        )

        fraud_count = min(
            fraud_count,
            len(working_transactions),
        )

        scenario_counts = self._empty_scenario_counts()

        if fraud_count == 0:
            return FraudInjectionResult(
                transactions=working_transactions,
                scenario_counts=scenario_counts,
            )

        available_indices = list(
            range(len(working_transactions))
        )

        for _ in range(fraud_count):
            if not available_indices:
                break

            transaction_index = self.random.choice(
                available_indices
            )
            available_indices.remove(
                transaction_index
            )

            transaction = working_transactions[
                transaction_index
            ]

            scenario = self._choose_scenario(
                devices=devices,
            )

            modified_transaction = self._apply_scenario(
                scenario=scenario,
                customer=customer,
                transaction=transaction,
                transactions=working_transactions,
                devices=devices,
            )

            working_transactions[
                transaction_index
            ] = modified_transaction

            scenario_counts[scenario] += 1

        working_transactions.sort(
            key=lambda transaction: transaction.transaction_timestamp
        )

        return FraudInjectionResult(
            transactions=working_transactions,
            scenario_counts=scenario_counts,
        )

    def _choose_scenario(
        self,
        devices: Sequence[DeviceState],
    ) -> str:
        """
        Select a fraud scenario using configured relative weights.

        Scenarios requiring unavailable data are excluded. In particular,
        new-device fraud requires at least one secondary device.

        Parameters
        ----------
        devices:
            Customer devices available to the simulation.

        Returns
        -------
        str
            Selected scenario name.
        """

        scenarios: list[str] = []
        weights: list[float] = []

        if len(devices) >= 2:
            scenarios.append("new_device")
            weights.append(
                self.config.new_device_weight
            )

        scenarios.extend(
            [
                "geographic_anomaly",
                "amount_anomaly",
                "velocity",
            ]
        )

        weights.extend(
            [
                self.config.geographic_anomaly_weight,
                self.config.amount_anomaly_weight,
                self.config.velocity_weight,
            ]
        )

        return self.random.choices(
            scenarios,
            weights=weights,
            k=1,
        )[0]

    def _apply_scenario(
        self,
        scenario: str,
        customer: CustomerState,
        transaction: TransactionState,
        transactions: Sequence[TransactionState],
        devices: Sequence[DeviceState],
    ) -> TransactionState:
        """
        Apply one synthetic fraud scenario to a transaction.

        Parameters
        ----------
        scenario:
            Name of the synthetic scenario to apply.
        customer:
            Customer associated with the transaction.
        transaction:
            Transaction selected for modification.
        transactions:
            Other transactions belonging to the customer.
        devices:
            Customer devices available to the simulation.

        Returns
        -------
        TransactionState
            Modified transaction with ``is_fraud=True``.
        """

        if scenario == "new_device":
            return self._apply_new_device_scenario(
                customer=customer,
                transaction=transaction,
                devices=devices,
            )

        if scenario == "geographic_anomaly":
            return self._apply_geographic_anomaly(
                customer=customer,
                transaction=transaction,
            )

        if scenario == "amount_anomaly":
            return self._apply_amount_anomaly(
                transaction=transaction,
            )

        if scenario == "velocity":
            return self._apply_velocity_scenario(
                customer=customer,
                transaction=transaction,
                transactions=transactions,
            )

        raise ValueError(
            f"Unknown fraud scenario: {scenario}"
        )

    def _apply_new_device_scenario(
        self,
        customer: CustomerState,
        transaction: TransactionState,
        devices: Sequence[DeviceState],
    ) -> TransactionState:
        """
        Create a transaction shortly after a secondary device appears.

        Parameters
        ----------
        customer:
            Customer associated with the transaction.
        transaction:
            Baseline transaction to modify.
        devices:
            Customer devices.

        Returns
        -------
        TransactionState
            Fraud-labelled transaction using a recently introduced device.
        """

        secondary_devices = [
            device
            for device in devices
            if device.first_seen > customer.customer_start_time
            and device.first_seen < customer.customer_end_time
        ]

        if not secondary_devices:
            return self._apply_amount_anomaly(
                transaction=transaction,
            )

        device = max(
            secondary_devices,
            key=lambda candidate: candidate.first_seen,
        )

        maximum_offset = min(
            timedelta(
                days=self.config.new_device_window_days
            ),
            customer.customer_end_time
            - device.first_seen,
        )

        maximum_seconds = max(
            0.0,
            maximum_offset.total_seconds(),
        )

        offset_seconds = 0.0

        if maximum_seconds > 0:
            offset_seconds = self.random.uniform(
                0.0,
                maximum_seconds,
            )

        transaction_timestamp = (
            device.first_seen
            + timedelta(seconds=offset_seconds)
        )

        transaction_timestamp = min(
            transaction_timestamp,
            customer.customer_end_time,
        )

        return replace(
            transaction,
            device_id=device.device_id,
            transaction_timestamp=transaction_timestamp,
            is_fraud=True,
        )

    def _apply_geographic_anomaly(
        self,
        customer: CustomerState,
        transaction: TransactionState,
    ) -> TransactionState:
        """
        Move a transaction to an unusual customer geography.

        Parameters
        ----------
        customer:
            Customer whose usual countries define normal geography.
        transaction:
            Baseline transaction to modify.

        Returns
        -------
        TransactionState
            Fraud-labelled transaction in an unusual country.
        """

        unusual_countries = [
            country
            for country in COUNTRIES
            if country not in customer.usual_countries
        ]

        if not unusual_countries:
            return self._apply_amount_anomaly(
                transaction=transaction,
            )

        country = self.random.choice(
            unusual_countries
        )

        ip_address = self._generate_synthetic_ip(
            transaction_id=transaction.transaction_id,
            country=country,
        )

        return replace(
            transaction,
            transaction_country=country,
            ip_address=ip_address,
            is_fraud=True,
        )

    def _apply_amount_anomaly(
        self,
        transaction: TransactionState,
    ) -> TransactionState:
        """
        Increase a transaction amount above the normal baseline.

        Parameters
        ----------
        transaction:
            Baseline transaction to modify.

        Returns
        -------
        TransactionState
            Fraud-labelled transaction with an anomalously large amount.
        """

        amount = round(
            max(
                0.50,
                transaction.amount
                * self.config.amount_anomaly_multiplier,
            ),
            2,
        )

        return replace(
            transaction,
            amount=amount,
            is_fraud=True,
        )

    def _apply_velocity_scenario(
        self,
        customer: CustomerState,
        transaction: TransactionState,
        transactions: Sequence[TransactionState],
    ) -> TransactionState:
        """
        Move a transaction close to another transaction in time.

        Parameters
        ----------
        customer:
            Customer defining the valid lifecycle boundaries.
        transaction:
            Transaction selected for modification.
        transactions:
            Customer transaction set from which a temporal anchor is chosen.

        Returns
        -------
        TransactionState
            Fraud-labelled transaction forming a short temporal burst.
        """

        other_transactions = [
            candidate
            for candidate in transactions
            if candidate.transaction_id
            != transaction.transaction_id
        ]

        if not other_transactions:
            return self._apply_amount_anomaly(
                transaction=transaction,
            )

        anchor = min(
            other_transactions,
            key=lambda candidate: abs(
                (
                    candidate.transaction_timestamp
                    - transaction.transaction_timestamp
                ).total_seconds()
            ),
        )

        maximum_seconds = max(
            1,
            self.config.velocity_window_minutes * 60 - 1,
        )

        offset_seconds = self.random.randint(
            1,
            maximum_seconds,
        )

        candidate_timestamp = (
            anchor.transaction_timestamp
            + timedelta(seconds=offset_seconds)
        )

        if (
            candidate_timestamp
            > customer.customer_end_time
        ):
            candidate_timestamp = (
                anchor.transaction_timestamp
                - timedelta(seconds=offset_seconds)
            )

        candidate_timestamp = max(
            customer.customer_start_time,
            min(
                candidate_timestamp,
                customer.customer_end_time,
            ),
        )

        return replace(
            transaction,
            transaction_timestamp=candidate_timestamp,
            is_fraud=True,
        )

    def _generate_synthetic_ip(
        self,
        transaction_id: uuid.UUID,
        country: str,
    ) -> IPv4Address:
        """
        Generate a stable synthetic IPv4 address.

        Parameters
        ----------
        transaction_id:
            Transaction identifier used as part of the stable input.
        country:
            Simulated transaction country.

        Returns
        -------
        IPv4Address
            Stable synthetic IPv4 address.
        """

        digest = hashlib.sha256(
            f"fraud-{transaction_id}-{country}".encode(
                "utf-8"
            )
        ).digest()

        numeric_value = int.from_bytes(
            digest[:4],
            byteorder="big",
        )

        return IPv4Address(numeric_value)

    def _empty_scenario_counts(
        self,
    ) -> dict[str, int]:
        """
        Create an empty scenario-count mapping.

        Returns
        -------
        dict[str, int]
            Scenario names initialized to zero.
        """

        return {
            "new_device": 0,
            "geographic_anomaly": 0,
            "amount_anomaly": 0,
            "velocity": 0,
        }

    def _validate_configuration(self) -> None:
        """
        Validate fraud-generation configuration.

        Raises
        ------
        ValueError
            If fraud rate or scenario weights are invalid.
        """

        if not 0.0 <= self.config.fraud_rate <= 1.0:
            raise ValueError(
                "fraud_rate must be between 0.0 and 1.0"
            )

        weights = [
            self.config.new_device_weight,
            self.config.geographic_anomaly_weight,
            self.config.amount_anomaly_weight,
            self.config.velocity_weight,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Fraud scenario weights cannot be negative."
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one fraud scenario weight must be positive."
            )