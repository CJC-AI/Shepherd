"""
Validate the precise semantics of Shepherd's historical fraud features.

This test uses a small deterministic transaction sequence where the expected
historical feature values are known in advance. It verifies that the feature
engine excludes the current transaction from historical aggregates and
correctly applies temporal lookback windows.
"""

from datetime import datetime, timedelta, timezone
from ipaddress import IPv4Address
from uuid import uuid4

from database.models import Transaction
from features.feature_engine import (
    FeatureEngine,
    FeatureReferenceData,
    MerchantReference,
)


def build_transaction(
    *,
    transaction_id,
    account_id,
    merchant_id,
    device_id,
    timestamp,
    amount,
):
    """
    Build a minimal transaction ORM object for feature testing.

    Parameters
    ----------
    transaction_id:
        Transaction identifier.
    account_id:
        Account identifier.
    merchant_id:
        Merchant identifier.
    device_id:
        Device identifier.
    timestamp:
        Transaction timestamp.
    amount:
        Transaction amount.

    Returns
    -------
    Transaction
        In-memory transaction object.
    """

    return Transaction(
        transaction_id=transaction_id,
        account_id=account_id,
        merchant_id=merchant_id,
        device_id=device_id,
        amount=amount,
        currency="USD",
        transaction_timestamp=timestamp,
        transaction_country="NG",
        ip_address=IPv4Address("192.0.2.1"),
        is_fraud=False,
    )


def main() -> None:
    """
    Validate historical feature semantics on a controlled sequence.

    Returns
    -------
    None
    """

    customer_id = uuid4()
    account_id = uuid4()
    merchant_id = uuid4()
    device_id = uuid4()

    start = datetime(
        2026,
        6,
        1,
        12,
        0,
        tzinfo=timezone.utc,
    )

    transactions = [
        build_transaction(
            transaction_id=uuid4(),
            account_id=account_id,
            merchant_id=merchant_id,
            device_id=device_id,
            timestamp=start,
            amount=100.0,
        ),
        build_transaction(
            transaction_id=uuid4(),
            account_id=account_id,
            merchant_id=merchant_id,
            device_id=device_id,
            timestamp=start + timedelta(minutes=5),
            amount=200.0,
        ),
        build_transaction(
            transaction_id=uuid4(),
            account_id=account_id,
            merchant_id=merchant_id,
            device_id=device_id,
            timestamp=start + timedelta(minutes=20),
            amount=50.0,
        ),
    ]

    references = FeatureReferenceData(
        account_to_customer={
            account_id: customer_id,
        },
        customer_home_country={
            customer_id: "NG",
        },
        device_first_seen={
            device_id: start - timedelta(hours=1),
        },
        merchant_reference={
            merchant_id: MerchantReference(
                merchant_category="groceries",
                risk_score=0.1,
            ),
        },
    )

    engine = FeatureEngine(
        references=references,
    )

    features = list(
        engine.compute(transactions)
    )

    first = features[0]
    second = features[1]
    third = features[2]

    assert first.transactions_before == 0
    assert first.tx_count_10m == 0
    assert first.avg_amount_before == 0.0
    assert first.amount_vs_avg == 0.0

    assert second.transactions_before == 1
    assert second.tx_count_10m == 1
    assert second.avg_amount_before == 100.0
    assert second.amount_vs_avg == 2.0

    assert third.transactions_before == 2
    assert third.tx_count_10m == 0
    assert third.avg_amount_before == 150.0
    assert third.amount_vs_avg == (
        50.0 / 150.0
    )

    print("feature semantics test: PASS")


if __name__ == "__main__":
    main()