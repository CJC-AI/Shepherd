"""
Generate and persist the canonical device population.

This module generates devices for the customers that already exist in the
Shepherd PostgreSQL database and persists the resulting DeviceState objects.

The module deliberately separates database reads from device generation so
that the simulation logic remains independent from SQLAlchemy.
"""

from sqlalchemy import select

from database.models import Customer
from database.session import SessionLocal
from simulation.customer_state import CustomerState
from simulation.device_generator import DeviceGenerator


def load_customer_ids() -> list:
    """
    Load customer identifiers from PostgreSQL.

    Returns
    -------
    list
        Customer UUIDs currently stored in the customers table.
    """
    with SessionLocal() as session:
        return list(session.scalars(select(Customer.customer_id)).all())


def generate_devices_for_customers(
    customers: list[CustomerState],
    seed: int | None = None,
):
    """
    Generate DeviceState objects for simulated customers.

    Parameters
    ----------
    customers:
        CustomerState objects whose devices should be generated.
    seed:
        Optional random seed used for reproducible device timestamps.

    Returns
    -------
    list[DeviceState]
        Generated device states.
    """
    generator = DeviceGenerator(seed=seed)

    return [
        device
        for customer in customers
        for device in generator.generate_for_customer(customer)
    ]