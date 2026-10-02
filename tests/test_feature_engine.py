"""
Validate leakage-safe feature computation against persisted Shepherd data.

This test loads reference data from PostgreSQL, selects one customer's
transactions in chronological order, computes features, and verifies
important temporal invariants.
"""

from sqlalchemy import select

from database.models import (
    Account,
    Customer,
    Device,
    Merchant,
    Transaction,
)
from database.session import SessionLocal
from features.feature_engine import (
    FeatureEngine,
    FeatureReferenceData,
    MerchantReference,
)


def main() -> None:
    """
    Compute features for one customer's persisted transactions.

    Returns
    -------
    None
    """

    with SessionLocal() as session:
        customer = session.scalars(
            select(Customer).limit(1)
        ).first()

        if customer is None:
            raise RuntimeError(
                "No customers were found in the database."
            )

        customer_id = customer.customer_id

        account_to_customer = {
            account.account_id: account.customer_id
            for account in session.scalars(
                select(Account)
            )
        }

        customer_home_country = {
            customer.customer_id: customer.signup_country
            for customer in session.scalars(
                select(Customer)
            )
        }

        device_first_seen = {
            device.device_id: device.first_seen
            for device in session.scalars(
                select(Device)
            )
        }

        merchant_reference = {
            merchant.merchant_id: MerchantReference(
                merchant_category=merchant.merchant_category,
                risk_score=merchant.risk_score,
            )
            for merchant in session.scalars(
                select(Merchant)
            )
        }

        transactions = session.scalars(
            select(Transaction)
            .join(
                Account,
                Transaction.account_id
                == Account.account_id,
            )
            .where(
                Account.customer_id == customer_id
            )
            .order_by(
                Transaction.transaction_timestamp,
                Transaction.transaction_id,
            )
        ).all()

        references = FeatureReferenceData(
            account_to_customer=account_to_customer,
            customer_home_country=customer_home_country,
            device_first_seen=device_first_seen,
            merchant_reference=merchant_reference,
        )

        engine = FeatureEngine(
            references=references
        )

        features = list(
            engine.compute(transactions)
        )

    print("customer:", customer_id)
    print("transactions:", len(transactions))
    print("features:", len(features))

    assert len(features) == len(
        transactions
    )

    assert features[0].transactions_before == 0
    assert features[0].tx_count_10m == 0
    assert features[0].tx_count_1h == 0
    assert features[0].tx_count_24h == 0
    assert (
        features[0].seconds_since_previous_tx
        is None
    )

    transaction_timestamps = [
        transaction.transaction_timestamp
        for transaction in transactions
    ]

    assert transaction_timestamps == sorted(
        transaction_timestamps
    )

    for previous, current in zip(
        features,
        features[1:],
    ):
        assert (
            current.transaction_timestamp
            >= previous.transaction_timestamp
        )

        assert (
            current.transactions_before
            >= previous.transactions_before
        )

    print("first feature row:", features[0])
    print("last feature row:", features[-1])
    print("feature computation test: PASS")


if __name__ == "__main__":
    main()