"""
Build and persist the canonical Shepherd simulation population.

This script generates the configured customer population together with its
accounts and devices, then persists all entities to PostgreSQL in dependency
order.
"""

from simulation.population_builder import PopulationBuilder
from simulation.population_repository import save_population
from simulation.simulation_config import DEFAULT_SIMULATION_CONFIG


def main() -> None:
    """
    Generate and persist the canonical simulation population.

    Returns
    -------
    None
    """
    population = PopulationBuilder(
        DEFAULT_SIMULATION_CONFIG,
        seed=42,
    ).build()

    print("Generated population:")
    print("  customers:", len(population.customers))
    print("  accounts:", len(population.accounts))
    print("  devices:", len(population.devices))

    save_population(population)

    print("Population persistence: COMPLETE")


if __name__ == "__main__":
    main()