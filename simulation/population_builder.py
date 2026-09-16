"""
Build the initial Shepherd simulation population.

This module orchestrates generation of customers, accounts, and devices while
keeping the simulation state in memory. Persistence is intentionally handled
after generation so that downstream generators can use the complete
CustomerState produced by the simulation.
"""

from dataclasses import dataclass

from simulation.account_generator import AccountGenerator
from simulation.account_state import AccountState
from simulation.device_generator import DeviceGenerator
from simulation.device_state import DeviceState
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from simulation.customer_state import CustomerState


@dataclass
class Population:
    """
    Container holding the generated Shepherd simulation population.

    Attributes
    ----------
    customers:
        Simulated customer states.
    accounts:
        Simulated account states belonging to the generated customers.
    devices:
        Simulated device states belonging to the generated customers.
    """

    customers: list[CustomerState]
    accounts: list[AccountState]
    devices: list[DeviceState]


class PopulationBuilder:
    """
    Build a complete simulation population.

    The builder generates customers first because accounts and devices depend
    on customer state. The same CustomerState objects are then passed into
    both downstream generators so that relationships and lifecycle information
    remain consistent.

    Parameters
    ----------
    config:
        Simulation configuration defining the simulation period and population
        size.
    seed:
        Optional base seed used to make the generated population reproducible.
    """

    def __init__(
        self,
        config: SimulationConfig,
        seed: int | None = None,
    ):
        """
        Initialize the population builder.

        Parameters
        ----------
        config:
            Simulation configuration.
        seed:
            Optional base random seed.
        """
        self.config = config
        self.seed = seed

    def build(self) -> Population:
        """
        Generate customers, accounts, and devices.

        Returns
        -------
        Population
            Complete in-memory simulation population.
        """
        runner = SimulationRunner(
            config=self.config,
            seed=self.seed,
        )

        customers = runner.generate_customers()

        account_generator = AccountGenerator(
            seed=self._derive_seed(1),
        )

        accounts = [
            account
            for customer in customers
            for account in account_generator.generate_for_customer(
                customer
            )
        ]

        device_generator = DeviceGenerator(
            seed=self._derive_seed(2),
        )

        devices = [
            device
            for customer in customers
            for device in device_generator.generate_for_customer(
                customer
            )
        ]

        return Population(
            customers=customers,
            accounts=accounts,
            devices=devices,
        )

    def _derive_seed(self, offset: int) -> int | None:
        """
        Derive a deterministic child seed from the base seed.

        Parameters
        ----------
        offset:
            Integer offset used to produce a distinct seed for a downstream
            generator.

        Returns
        -------
        int | None
            Derived deterministic seed, or None when no base seed was
            supplied.
        """
        if self.seed is None:
            return None

        return self.seed + offset