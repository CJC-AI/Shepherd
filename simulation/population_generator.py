from simulation.config import (
    ARCHETYPE_BY_NAME,
    ARCHETYPE_WEIGHTS,
    CustomerArchetype,
)
from simulation.customer_generator import CustomerGenerator
from simulation.customer_state import CustomerState
from simulation.clock import SimulationClock


class PopulationGenerator:
    def __init__(self, clock: SimulationClock, seed: int | None = None):
        self.customer_generator = CustomerGenerator(clock=clock, seed=seed)

    def generate(
        self,
        n_customers: int,
    ) -> list[CustomerState]:
        if n_customers <= 0:
            raise ValueError("n_customers must be greater than zero")

        archetype_names = list(ARCHETYPE_WEIGHTS.keys())
        weights = list(ARCHETYPE_WEIGHTS.values())

        customers: list[CustomerState] = []

        for _ in range(n_customers):
            archetype_name = self.customer_generator.random.choices(
                archetype_names,
                weights=weights,
                k=1,
            )[0]

            archetype = ARCHETYPE_BY_NAME[archetype_name]

            customer = self.customer_generator.generate(
                archetype
            )

            customers.append(customer)

        return customers

    