"""
Build and persist the canonical Shepherd merchant population.

This script generates the deterministic merchant catalog using the configured
simulation seed and persists the resulting MerchantState objects to
PostgreSQL.
"""

from simulation.merchant_generator import MerchantGenerator
from simulation.merchant_repository import save_merchants


MERCHANT_COUNT = 500
MERCHANT_SEED = 42


def main() -> None:
    """
    Generate and persist the canonical merchant catalog.

    Returns
    -------
    None
    """

    merchants = MerchantGenerator(
        seed=MERCHANT_SEED,
    ).generate(
        count=MERCHANT_COUNT,
    )

    print("Generated merchants:")
    print("  merchants:", len(merchants))

    saved = save_merchants(merchants)

    print("Persisted merchants:", saved)

    if saved != len(merchants):
        raise RuntimeError(
            f"Expected to persist {len(merchants)} merchants, "
            f"but persisted {saved}."
        )

    print("Merchant population persistence: COMPLETE")


if __name__ == "__main__":
    main()