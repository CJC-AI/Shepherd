"""
Generate simulated transactions in bounded batches.

This module orchestrates the transaction generator and fraud scenario engine
over the canonical Shepherd population without retaining the entire
transaction dataset in memory.

Customers, accounts, and devices are read from the already-generated
Population object. Transactions are produced customer-by-customer and yielded
in configurable batches suitable for database persistence.
"""

from collections import defaultdict
from collections.abc import Iterator

from simulation.customer_state import CustomerState
from simulation.device_state import DeviceState
from simulation.fraud_config import (
    DEFAULT_FRAUD_SCENARIO_CONFIG,
    FraudScenarioConfig,
)
from simulation.merchant_state import MerchantState
from simulation.population_builder import Population
from simulation.transaction_config import (
    DEFAULT_TRANSACTION_CONFIG,
    TransactionSimulationConfig,
)
from simulation.transaction_generator import TransactionGenerator
from simulation.transaction_state import TransactionState
from simulation.fraud_scenario_engine import FraudScenarioEngine


class TransactionPopulationBuilder:
    """
    Generate the complete transaction population in bounded batches.

    Parameters
    ----------
    population:
        Canonical customer, account, and device population.
    merchants:
        Canonical merchant catalog.
    transaction_config:
        Configuration controlling normal transaction generation.
    fraud_config:
        Configuration controlling synthetic fraud injection.
    transaction_seed:
        Random seed used for reproducible transaction generation.
    fraud_seed:
        Random seed used for reproducible fraud scenario generation.
    """

    def __init__(
        self,
        population: Population,
        merchants: list[MerchantState],
        transaction_config: TransactionSimulationConfig = (
            DEFAULT_TRANSACTION_CONFIG
        ),
        fraud_config: FraudScenarioConfig = (
            DEFAULT_FRAUD_SCENARIO_CONFIG
        ),
        transaction_seed: int | None = 46,
        fraud_seed: int | None = 47,
    ):
        """
        Initialize the transaction population builder.

        Parameters
        ----------
        population:
            Canonical customer, account, and device population.
        merchants:
            Merchant catalog available to all customers.
        transaction_config:
            Normal transaction-generation configuration.
        fraud_config:
            Synthetic fraud-generation configuration.
        transaction_seed:
            Optional seed controlling transaction generation.
        fraud_seed:
            Optional seed controlling fraud injection.
        """

        self.population = population
        self.merchants = merchants

        self.transaction_generator = TransactionGenerator(
            config=transaction_config,
            seed=transaction_seed,
        )

        self.fraud_engine = FraudScenarioEngine(
            config=fraud_config,
            seed=fraud_seed,
        )

        self.accounts_by_customer = (
            self._group_accounts_by_customer()
        )

        self.devices_by_customer = (
            self._group_devices_by_customer()
        )

    def generate_batches(
        self,
        batch_size: int = 5000,
    ) -> Iterator[list[TransactionState]]:
        """
        Generate transactions as bounded batches.

        Transactions are generated one customer at a time. The resulting
        transactions are added to an in-memory buffer until batch_size is
        reached, then yielded to the caller. The buffer is cleared before
        processing continues.

        Parameters
        ----------
        batch_size:
            Maximum number of transactions yielded in each batch.

        Yields
        ------
        list[TransactionState]
            Bounded batch of generated and fraud-labelled transactions.

        Raises
        ------
        ValueError
            If batch_size is less than or equal to zero.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        batch: list[TransactionState] = []

        for customer in self.population.customers:
            customer_accounts = (
                self.accounts_by_customer.get(
                    customer.customer_id,
                    [],
                )
            )

            customer_devices = (
                self.devices_by_customer.get(
                    customer.customer_id,
                    [],
                )
            )

            baseline_transactions = (
                self.transaction_generator
                .generate_for_customer(
                    customer=customer,
                    accounts=customer_accounts,
                    devices=customer_devices,
                    merchants=self.merchants,
                )
            )

            result = self.fraud_engine.apply_for_customer(
                customer=customer,
                transactions=baseline_transactions,
                devices=customer_devices,
            )

            batch.extend(result.transactions)

            while len(batch) >= batch_size:
                yield batch[:batch_size]
                batch = batch[batch_size:]

        if batch:
            yield batch

    def _group_accounts_by_customer(
        self,
    ) -> dict:
        """
        Group account states by owning customer.

        Returns
        -------
        dict
            Mapping from customer UUID to AccountState objects.
        """

        accounts_by_customer = defaultdict(list)

        for account in self.population.accounts:
            accounts_by_customer[
                account.customer_id
            ].append(account)

        return dict(accounts_by_customer)

    def _group_devices_by_customer(
        self,
    ) -> dict:
        """
        Group device states by owning customer.

        Returns
        -------
        dict
            Mapping from customer UUID to DeviceState objects.
        """

        devices_by_customer = defaultdict(list)

        for device in self.population.devices:
            devices_by_customer[
                device.customer_id
            ].append(device)

        return dict(devices_by_customer)