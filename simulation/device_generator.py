"""
Generate device lifecycle state for simulated bank customers.

This module converts the device identifiers stored on a CustomerState
into DeviceState objects while assigning simulated first-seen timestamps.

The generator intentionally keeps device creation separate from customer
creation so that device onboarding can evolve independently over the
customer's simulated lifetime.
"""

from random import Random

from simulation.customer_state import CustomerState
from simulation.device_state import DeviceState
from simulation.time_utils import random_datetime


class DeviceGenerator:
    """
    Generate simulated devices for customers.

    A customer's first device becomes active at the customer's start time.
    Additional devices are assigned first-seen timestamps later in the
    customer's lifecycle, allowing the simulation to represent device
    onboarding and device-age differences.

    Parameters
    ----------
    seed:
        Optional random seed used to make generated device timing
        reproducible.
    """

    def __init__(self, seed: int | None = None):
        self.random = Random(seed)

    def generate_for_customer(
        self,
        customer: CustomerState,
    ) -> list[DeviceState]:
        """
        Generate DeviceState objects for one customer.

        The customer's first known device is treated as the primary
        device and is activated at the customer's start time. Remaining
        devices are assigned random first-seen timestamps between the
        customer's start time and the end of the customer's possible
        lifecycle window.

        Parameters
        ----------
        customer:
            CustomerState containing the customer's known device IDs
            and simulated start time.

        Returns
        -------
        list[DeviceState]
            Generated device states belonging to the customer.
        """

        if not customer.known_devices:
            return []

        devices: list[DeviceState] = []

        primary_device_id = customer.known_devices[0]

        devices.append(
            DeviceState(
                device_id=primary_device_id,
                customer_id=customer.customer_id,
                device_trust_score=0.8,
                first_seen=customer.customer_start_time,
            )
        )

        customer.active_device_ids = [
            primary_device_id
        ]

        for device_id in customer.known_devices[1:]:
            devices.append(
                DeviceState(
                    device_id=device_id,
                    customer_id=customer.customer_id,
                    device_trust_score=0.5,
                    first_seen=customer.customer_start_time,
                )
            )

        return devices