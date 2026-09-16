"""
Generate simulated bank customer state.

This module produces CustomerState objects containing both behavioral
attributes and simulation lifecycle information.

Entity identifiers are generated deterministically from stable simulation
identifiers so that rebuilding a population with the same configuration
produces the same customer and device UUIDs.
"""

import random
import uuid
from datetime import datetime

from simulation.clock import SimulationClock
from simulation.config import CustomerArchetype
from simulation.constants import (
    ARCHETYPE_MERCHANT_CATEGORIES,
    COUNTRIES,
    TRAVEL_DESTINATIONS,
)
from simulation.customer_state import CustomerState


SIMULATION_NAMESPACE = uuid.UUID(
    "7f4a7c0e-0dd8-4c0d-9b48-4a5f2e2e7c01"
)


class CustomerGenerator:
    """
    Generate deterministic simulated customers.

    Parameters
    ----------
    clock:
        Simulation clock used when explicit lifecycle timestamps are not
        supplied.
    seed:
        Optional seed controlling stochastic customer attributes.
    """

    def __init__(
        self,
        clock: SimulationClock,
        seed: int | None = None,
    ):
        """
        Initialize the customer generator.

        Parameters
        ----------
        clock:
            Simulation clock used as the default lifecycle timestamp source.
        seed:
            Optional random seed used for reproducible behavioral attributes.
        """
        self.clock = clock
        self.random = random.Random(seed)

    def generate(
        self,
        archetype: CustomerArchetype,
        customer_index: int | None = None,
        customer_start_time: datetime | None = None,
        customer_end_time: datetime | None = None,
    ) -> CustomerState:
        """
        Generate one simulated customer.

        Parameters
        ----------
        archetype:
            Behavioral archetype assigned to the customer.
        customer_index:
            Stable position of the customer within the simulation population.
            When supplied, it is used to generate deterministic customer and
            device UUIDs.
        customer_start_time:
            Optional timestamp at which the customer's simulated lifecycle
            begins. When omitted, the simulation clock's current time is used.
        customer_end_time:
            Optional timestamp at which the customer's simulated lifecycle
            ends. When omitted, the resolved start time is used.

        Returns
        -------
        CustomerState
            Fully populated simulated customer state.

        Raises
        ------
        ValueError
            If the end time occurs before the start time.
            If customer_index is negative.
        """

        resolved_start_time = (
            customer_start_time
            if customer_start_time is not None
            else self.clock.now()
        )

        resolved_end_time = (
            customer_end_time
            if customer_end_time is not None
            else resolved_start_time
        )

        if resolved_end_time < resolved_start_time:
            raise ValueError(
                "customer_end_time must be greater than or equal to "
                "customer_start_time"
            )

        if customer_index is not None and customer_index < 0:
            raise ValueError("customer_index must be non-negative")

        if customer_index is None:
            customer_id = uuid.uuid4()
        else:
            customer_id = uuid.uuid5(
                SIMULATION_NAMESPACE,
                f"customer-{customer_index}",
            )

        home_country = self.random.choice(COUNTRIES)

        average_spend = max(
            1.0,
            self.random.gauss(
                archetype.average_spend,
                archetype.average_spend * 0.20,
            ),
        )

        spend_std = max(
            1.0,
            self.random.gauss(
                archetype.spend_std,
                archetype.spend_std * 0.15,
            ),
        )

        transactions_per_day = self.random.randint(
            archetype.transactions_per_day_min,
            archetype.transactions_per_day_max,
        )

        device_count = self.random.randint(
            archetype.device_count_min,
            archetype.device_count_max,
        )

        if customer_index is None:
            known_devices = [
                uuid.uuid4()
                for _ in range(device_count)
            ]
        else:
            known_devices = [
                uuid.uuid5(
                    customer_id,
                    f"device-{device_index}",
                )
                for device_index in range(device_count)
            ]

        usual_countries = self._generate_usual_countries(
            home_country=home_country,
            archetype=archetype,
        )

        preferred_categories = self._generate_preferred_categories(
            archetype
        )

        return CustomerState(
            customer_id=customer_id,
            archetype=archetype,
            customer_start_time=resolved_start_time,
            customer_end_time=resolved_end_time,
            home_country=home_country,
            average_spend=round(average_spend, 2),
            spend_std=round(spend_std, 2),
            transactions_per_day=transactions_per_day,
            usual_countries=usual_countries,
            known_devices=known_devices,
            preferred_merchant_categories=preferred_categories,
        )

    def _generate_usual_countries(
        self,
        home_country: str,
        archetype: CustomerArchetype,
    ) -> list[str]:
        """
        Generate the countries a customer commonly uses.

        Parameters
        ----------
        home_country:
            Customer's primary country.
        archetype:
            Customer behavioral archetype.

        Returns
        -------
        list[str]
            Unique list containing the home country and, when applicable,
            likely international destinations.
        """

        countries = [home_country]

        if archetype.international_probability > 0.10:
            destinations = TRAVEL_DESTINATIONS.get(
                home_country,
                COUNTRIES,
            )

            additional_count = min(3, len(destinations))

            if additional_count > 0:
                countries.extend(
                    self.random.sample(
                        destinations,
                        k=additional_count,
                    )
                )

        return list(dict.fromkeys(countries))

    def _generate_preferred_categories(
        self,
        archetype: CustomerArchetype,
    ) -> list[str]:
        """
        Generate merchant categories preferred by the customer.

        Parameters
        ----------
        archetype:
            Customer behavioral archetype.

        Returns
        -------
        list[str]
            Sampled preferred merchant categories.
        """

        categories = ARCHETYPE_MERCHANT_CATEGORIES[
            archetype.name
        ]

        count = min(4, len(categories))

        return self.random.sample(
            categories,
            k=count,
        )

    def choose_transaction_country(
        self,
        customer: CustomerState,
    ) -> str:
        """
        Choose the country for a simulated transaction.

        Parameters
        ----------
        customer:
            Customer whose transaction location is being generated.

        Returns
        -------
        str
            Country code selected for the transaction.
        """

        international = (
            self.random.random()
            < customer.archetype.international_probability
        )

        if not international:
            return customer.home_country

        international_countries = [
            country
            for country in customer.usual_countries
            if country != customer.home_country
        ]

        if not international_countries:
            return customer.home_country

        return self.random.choice(international_countries)