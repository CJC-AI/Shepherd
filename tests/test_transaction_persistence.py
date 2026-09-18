"""
Verify transaction persistence and database cleanup.

This test recreates one deterministic customer from the canonical Shepherd
simulation, generates that customer's accounts and devices using the same
generation seeds used by the population builder, generates the canonical
merchant catalog, creates a small baseline transaction set, persists the
transactions, verifies them in PostgreSQL, and removes the test transactions
when the test finishes.

The test does not create new customers, accounts, devices, or merchants.
It uses the already-persisted canonical reference population.
"""

from sqlalchemy import delete, select

from database.models import Transaction
from database.session import SessionLocal
from simulation.account_generator import AccountGenerator
from simulation.device_generator import DeviceGenerator
from simulation.merchant_generator import MerchantGenerator
from simulation.simulation_config import DEFAULT_SIMULATION_CONFIG
from simulation.simulation_runner import SimulationRunner
from simulation.transaction_generator import TransactionGenerator
from simulation.transaction_repository import save_transactions


def main() -> None:
    """
    Execute the transaction persistence test.

    Returns
    -------
    None
    """

    runner = SimulationRunner(
        DEFAULT_SIMULATION_CONFIG,
        seed=42,
    )

    customer = runner.generate_customers()[0]

    account_generator = AccountGenerator(
        seed=43,
    )

    accounts = account_generator.generate_for_customer(
        customer
    )

    device_generator = DeviceGenerator(
        seed=44,
    )

    devices = device_generator.generate_for_customer(
        customer
    )

    merchants = MerchantGenerator(
        seed=42,
    ).generate(500)

    transactions = TransactionGenerator(
        seed=46,
    ).generate_for_customer(
        customer=customer,
        accounts=accounts,
        devices=devices,
        merchants=merchants,
    )

    transaction_ids = [
        transaction.transaction_id
        for transaction in transactions
    ]

    try:
        saved = save_transactions(
            transactions
        )

        with SessionLocal() as session:
            persisted_transactions = session.scalars(
                select(Transaction).where(
                    Transaction.transaction_id.in_(
                        transaction_ids
                    )
                )
            ).all()

        print(
            "generated transactions:",
            len(transactions),
        )
        print(
            "saved transactions:",
            saved,
        )
        print(
            "persisted transactions:",
            len(persisted_transactions),
        )

        assert saved == len(transactions)
        assert (
            len(persisted_transactions)
            == len(transactions)
        )

        print("transaction persistence test: PASS")

    finally:
        with SessionLocal() as session:
            session.execute(
                delete(Transaction).where(
                    Transaction.transaction_id.in_(
                        transaction_ids
                    )
                )
            )
            session.commit()

        print("test data cleanup: COMPLETE")


if __name__ == "__main__":
    main()