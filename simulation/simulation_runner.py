import random

from simulation.clock import SimulationClock
from simulation.customer_generator import CustomerGenerator
from simulation.customer_state import CustomerState
from simulation.simulation_config import SimulationConfig
from simulation.time_utils import (
    datetime_at_progress,
    random_progress,
)


class SimulationRunner:
    def __init__(
        self,
        config: SimulationConfig,
        seed: int | None = None,
    ):
        self.config = config
        self.random = random.Random(seed)

        self.clock = SimulationClock(
            config.start_time
        )

        self.customer_generator = CustomerGenerator(
            clock=self.clock,
            seed=seed,
        )

    def generate_customers(self) -> list[CustomerState]:
        customers: list[CustomerState] = []

        for customer_index in range(self.config.customer_count):
            customer = self._generate_customer(
               customer_index=customer_index
            )

            customers.append(customer)

        return customers

    def _generate_customer(self, customer_index: int) -> CustomerState:
        from simulation.config import (
            ARCHETYPE_BY_NAME,
            ARCHETYPE_WEIGHTS,
        )

        archetype_names = list(
            ARCHETYPE_WEIGHTS.keys()
        )

        weights = list(
            ARCHETYPE_WEIGHTS.values()
        )

        archetype_name = self.random.choices(
            archetype_names,
            weights=weights,
            k=1,
        )[0]

        archetype = ARCHETYPE_BY_NAME[
            archetype_name
        ]

        progress = random_progress(
            self.random
        )

        customer_start_time = datetime_at_progress(
            self.config.start_time,
            self.config.end_time,
            progress,
        )

        customer_end_time = self.config.end_time

        return self.customer_generator.generate(
            archetype=archetype,
            customer_index=customer_index,
            customer_start_time=customer_start_time,
            customer_end_time=customer_end_time,
        )