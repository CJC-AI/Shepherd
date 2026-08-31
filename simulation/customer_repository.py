import hashlib

from database.models import Customer
from database.session import SessionLocal
from simulation.customer_state import CustomerState


def generate_email_hash(customer_id: str) -> str:
    return hashlib.sha256(
        customer_id.encode("utf-8")
    ).hexdigest()


def save_customers(
    customers: list[CustomerState],
    batch_size: int = 1000,
) -> int:
    if not customers:
        return 0

    total_saved = 0

    with SessionLocal() as session:
        for start in range(0, len(customers), batch_size):
            batch = customers[start:start + batch_size]

            customer_rows = [
                Customer(
                    customer_id=customer.customer_id,
                    email_hash=generate_email_hash(
                        str(customer.customer_id)
                    ),
                    signup_country=customer.home_country,
                    kyc_status="verified",
                    risk_rating=None,
                )
                for customer in batch
            ]

            session.add_all(customer_rows)
            session.flush()

            total_saved += len(customer_rows)

        session.commit()

    return total_saved