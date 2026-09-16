"""
Verify merchant persistence and cleanup.

This test generates a small deterministic merchant population, persists it
to PostgreSQL, verifies that the expected rows were written, and removes the
temporary records when the test finishes.
"""

from sqlalchemy import delete, select

from database.models import Merchant
from database.session import SessionLocal
from simulation.merchant_generator import MerchantGenerator
from simulation.merchant_repository import save_merchants


def main() -> None:
    """
    Execute the merchant persistence test.

    Returns
    -------
    None
    """

    merchants = MerchantGenerator(
        seed=42
    ).generate(10)

    merchant_ids = [
        merchant.merchant_id
        for merchant in merchants
    ]

    try:
        saved = save_merchants(merchants)

        with SessionLocal() as session:
            persisted_merchants = session.scalars(
                select(Merchant).where(
                    Merchant.merchant_id.in_(merchant_ids)
                )
            ).all()

        print("generated merchants:", len(merchants))
        print("saved merchants:", saved)
        print(
            "persisted merchants:",
            len(persisted_merchants),
        )

        assert saved == len(merchants)
        assert len(persisted_merchants) == len(merchants)

        print("merchant persistence test: PASS")

    finally:
        with SessionLocal() as session:
            session.execute(
                delete(Merchant).where(
                    Merchant.merchant_id.in_(merchant_ids)
                )
            )
            session.commit()

        print("test data cleanup: COMPLETE")


if __name__ == "__main__":
    main()