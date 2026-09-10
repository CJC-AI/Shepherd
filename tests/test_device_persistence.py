"""
Verify that simulated devices can be persisted to PostgreSQL.

This script generates a small customer population, creates their devices,
persists those devices, and verifies that the expected number of rows exists
in the database.
"""

from datetime import datetime, timezone

from sqlalchemy import select

from database.models import Device
from database.session import SessionLocal
from simulation.device_generator import DeviceGenerator
from simulation.device_repository import save_devices
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from simulation.customer_repository import save_customers


def main() -> None:
    """
    Generate and persist a small device population.

    The test uses five simulated customers so that database persistence
    can be validated without modifying the canonical 10,000-customer
    population.

    Returns
    -------
    None
    """
    config = SimulationConfig(
        start_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
        duration_days=90,
        customer_count=5,
    )

    runner = SimulationRunner(config, seed=42)
    customers = runner.generate_customers()
    save_customers(customers)

    device_generator = DeviceGenerator(seed=42)

    devices = [
        device
        for customer in customers
        for device in device_generator.generate_for_customer(customer)
    ]

    save_devices(devices)

    with SessionLocal() as session:
        persisted_devices = session.scalars(
            select(Device).where(
                Device.device_id.in_(
                    [device.device_id for device in devices]
                )
            )
        ).all()

    print("customers:", len(customers))
    print("generated devices:", len(devices))
    print("persisted devices:", len(persisted_devices))

    assert len(persisted_devices) == len(devices)

    print("persistence test: PASS")


if __name__ == "__main__":
    main()