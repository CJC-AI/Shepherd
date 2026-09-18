"""
Build and persist the canonical Shepherd transaction population.

This script reconstructs the deterministic reference population in memory,
generates the canonical merchant catalog, streams transactions customer by
customer, injects synthetic fraud scenarios, and persists transactions to
PostgreSQL in bounded batches.

The script intentionally refuses to start when the transactions table already
contains rows. This prevents accidental duplicate transaction insertion when
rerunning the canonical dataset build.
"""

from sqlalchemy import func, select

from database.models import Transaction
from database.session import SessionLocal
from simulation.merchant_generator import MerchantGenerator
from simulation.population_builder import PopulationBuilder
from simulation.simulation_config import DEFAULT_SIMULATION_CONFIG
from simulation.transaction_population import (
    TransactionPopulationBuilder,
)
from simulation.transaction_repository import save_transactions


MERCHANT_COUNT = 500
MERCHANT_SEED = 42
POPULATION_SEED = 42
TRANSACTION_SEED = 46
FRAUD_SEED = 47
TRANSACTION_BATCH_SIZE = 5000


def get_existing_transaction_count() -> int:
    """
    Return the number of persisted transactions.

    Returns
    -------
    int
        Number of rows currently present in the transactions table.
    """

    with SessionLocal() as session:
        return session.scalar(
            select(func.count()).select_from(Transaction)
        ) or 0


def main() -> None:
    """
    Generate and persist the canonical transaction population.

    The function verifies that the transaction table is empty, reconstructs
    the deterministic reference population and merchant catalog, streams
    transactions in bounded batches, persists each batch, and reports final
    generation statistics.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If transactions already exist in the database.
    """

    existing_count = get_existing_transaction_count()

    if existing_count != 0:
        raise RuntimeError(
            "Transaction table is not empty. "
            f"Found {existing_count} existing transactions. "
            "Refusing to generate the canonical dataset because doing so "
            "could create duplicate transaction IDs."
        )

    print("Building deterministic reference population...")

    population = PopulationBuilder(
        DEFAULT_SIMULATION_CONFIG,
        seed=POPULATION_SEED,
    ).build()

    merchants = MerchantGenerator(
        seed=MERCHANT_SEED,
    ).generate(
        MERCHANT_COUNT
    )

    print(
        "Reference population:",
        f"{len(population.customers):,} customers,",
        f"{len(population.accounts):,} accounts,",
        f"{len(population.devices):,} devices,",
        f"{len(merchants):,} merchants",
    )

    transaction_builder = TransactionPopulationBuilder(
        population=population,
        merchants=merchants,
        transaction_seed=TRANSACTION_SEED,
        fraud_seed=FRAUD_SEED,
    )

    total_transactions = 0
    total_fraud = 0
    batch_count = 0

    for batch in transaction_builder.generate_batches(
        batch_size=TRANSACTION_BATCH_SIZE,
    ):
        saved = save_transactions(batch)

        total_transactions += saved

        total_fraud += sum(
            transaction.is_fraud is True
            for transaction in batch
        )

        batch_count += 1

        if batch_count % 25 == 0:
            fraud_rate = (
                total_fraud / total_transactions
                if total_transactions
                else 0.0
            )

            print(
                f"Processed {batch_count:,} batches | "
                f"{total_transactions:,} transactions | "
                f"fraud={total_fraud:,} | "
                f"fraud_rate={fraud_rate:.4%}"
            )

    persisted_count = get_existing_transaction_count()

    print()
    print("Transaction population persistence: COMPLETE")
    print(f"Generated transactions: {total_transactions:,}")
    print(f"Fraud transactions:     {total_fraud:,}")
    print(f"Batches:                {batch_count:,}")
    print(f"Persisted transactions: {persisted_count:,}")

    if persisted_count != total_transactions:
        raise RuntimeError(
            "Persisted transaction count does not match the number "
            "reported by the repository."
        )


if __name__ == "__main__":
    main()