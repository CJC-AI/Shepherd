"""
Verify device persistence and database cleanup.

This test generates a small temporary customer population, persists the
customers and their devices, verifies that the expected devices were written
to PostgreSQL, and removes the temporary records when the test finishes.

The test intentionally uses a small population so that persistence can be
validated without modifying the canonical simulation dataset.
"""

from datetime import datetime, timezone

from sqlalchemy import delete, select

from database.models import Customer, Device
from database.session import SessionLocal
from simulation.customer_repository import save_customers
from simulation.device_generator import DeviceGenerator
from simulation.device_repository import save_devices
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner


def main() -> None:
    """
    Execute the device persistence test.

    A small deterministic population is generated, persisted, queried back
    from PostgreSQL, and then removed. Cleanup occurs in a finally block so
    that temporary test data is removed even when an assertion or database
    operation fails.

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

    device_generator = DeviceGenerator(seed=42)

    devices = [
        device
        for customer in customers
        for device in device_generator.generate_for_customer(customer)
    ]

    customer_ids = [customer.customer_id for customer in customers]
    device_ids = [device.device_id for device in devices]

    try:
        save_customers(customers)
        save_devices(devices)

        with SessionLocal() as session:
            persisted_devices = session.scalars(
                select(Device).where(
                    Device.device_id.in_(device_ids)
                )
            ).all()

        print("customers:", len(customers))
        print("generated devices:", len(devices))
        print("persisted devices:", len(persisted_devices))

        assert len(persisted_devices) == len(devices)

        print("persistence test: PASS")

    finally:
        with SessionLocal() as session:
            session.execute(
                delete(Customer).where(
                    Customer.customer_id.in_(customer_ids)
                )
            )
            session.commit()

        print("test data cleanup: COMPLETE")


if __name__ == "__main__":
    main()