"""
Validate streaming transaction population generation.

This test builds a small deterministic customer population and verifies that
transactions can be generated in bounded batches without retaining the full
transaction population in memory.
"""

from simulation.merchant_generator import MerchantGenerator
from simulation.population_builder import PopulationBuilder
from simulation.simulation_config import SimulationConfig
from simulation.transaction_population import (
    TransactionPopulationBuilder,
)
from simulation.simulation_config import DEFAULT_SIMULATION_CONFIG


def main() -> None:
    """
    Generate and validate a bounded transaction stream.

    Returns
    -------
    None
    """

    population = PopulationBuilder(
        DEFAULT_SIMULATION_CONFIG,
        seed=42,
    ).build()

    merchants = MerchantGenerator(
        seed=42,
    ).generate(500)

    builder = TransactionPopulationBuilder(
        population=population,
        merchants=merchants,
        transaction_seed=46,
        fraud_seed=47,
    )

    batch_count = 0
    transaction_count = 0
    fraud_count = 0
    largest_batch = 0

    for batch in builder.generate_batches(
        batch_size=5000,
    ):
        batch_count += 1
        transaction_count += len(batch)
        fraud_count += sum(
            transaction.is_fraud
            for transaction in batch
            if transaction.is_fraud is not None
        )

        largest_batch = max(
            largest_batch,
            len(batch),
        )

    print("customers:", len(population.customers))
    print("accounts:", len(population.accounts))
    print("devices:", len(population.devices))
    print("merchants:", len(merchants))
    print("batches:", batch_count)
    print("transactions:", transaction_count)
    print("fraud:", fraud_count)
    print("largest batch:", largest_batch)

    assert transaction_count > 0
    assert largest_batch <= 5000
    assert fraud_count > 0

    print("transaction population streaming test: PASS")


if __name__ == "__main__":
    main()