"""
Build the Shepherd offline fraud-feature dataset.

This module streams the persisted transaction history from PostgreSQL in
chronological order, computes leakage-safe features, and writes the resulting
feature rows to a Parquet dataset using bounded memory.

The generated Parquet dataset contains both model features and the fraud
ground-truth label required for supervised training. The fraud label is kept
as a target column and is never used by FeatureEngine when computing the
features.
"""

from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select

from database.models import Account, Transaction
from database.models import Customer, Device, Merchant
from database.session import SessionLocal
from features.feature_engine import (
    FeatureEngine,
    FeatureReferenceData,
    MerchantReference,
)


OUTPUT_PATH = Path(
    "artifacts/features/transaction_features.parquet"
)

TRANSACTION_BATCH_SIZE = 5_000
PROGRESS_INTERVAL = 100_000


FEATURE_SCHEMA = pa.schema(
    [
        pa.field(
            "transaction_id",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "transaction_timestamp",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "amount",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "merchant_risk_score",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "is_home_country",
            pa.bool_(),
            nullable=False,
        ),
        pa.field(
            "device_age_hours",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "transactions_before",
            pa.int64(),
            nullable=False,
        ),
        pa.field(
            "avg_amount_before",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "amount_vs_avg",
            pa.float64(),
            nullable=False,
        ),
        pa.field(
            "seconds_since_previous_tx",
            pa.float64(),
            nullable=True,
        ),
        pa.field(
            "tx_count_10m",
            pa.int64(),
            nullable=False,
        ),
        pa.field(
            "tx_count_1h",
            pa.int64(),
            nullable=False,
        ),
        pa.field(
            "tx_count_24h",
            pa.int64(),
            nullable=False,
        ),
        pa.field(
            "merchant_category_seen_before",
            pa.bool_(),
            nullable=False,
        ),
        pa.field(
            "is_fraud",
            pa.bool_(),
            nullable=False,
        ),
    ]
)


def load_reference_data() -> FeatureReferenceData:
    """
    Load the small reference datasets required by FeatureEngine.

    Returns
    -------
    FeatureReferenceData
        Account-to-customer, customer-country, device, and merchant mappings.

    Raises
    ------
    RuntimeError
        If required reference data is missing.
    """

    with SessionLocal() as session:
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

    if not account_to_customer:
        raise RuntimeError(
            "No account reference data was found."
        )

    if not customer_home_country:
        raise RuntimeError(
            "No customer reference data was found."
        )

    if not device_first_seen:
        raise RuntimeError(
            "No device reference data was found."
        )

    if not merchant_reference:
        raise RuntimeError(
            "No merchant reference data was found."
        )

    return FeatureReferenceData(
        account_to_customer=account_to_customer,
        customer_home_country=customer_home_country,
        device_first_seen=device_first_seen,
        merchant_reference=merchant_reference,
    )


def stream_transactions(
    batch_size: int,
) -> Iterator[Transaction]:
    """
    Stream persisted transactions in chronological order.

    Transactions are ordered globally by timestamp and then transaction ID.
    This ensures FeatureEngine receives transactions in causal order.

    Parameters
    ----------
    batch_size:
        Number of rows SQLAlchemy should fetch from PostgreSQL at a time.

    Yields
    ------
    Transaction
        Persisted transaction ORM objects.
    """

    statement = (
        select(Transaction)
        .order_by(
            Transaction.transaction_timestamp,
            Transaction.transaction_id,
        )
    )

    with SessionLocal() as session:
        result = session.scalars(
            statement
        ).yield_per(batch_size)

        for transaction in result:
            yield transaction


def feature_rows(
    transactions: Iterator[Transaction],
    engine: FeatureEngine,
) -> Iterator[dict]:
    """
    Convert streamed transactions into Parquet-ready feature rows.

    Parameters
    ----------
    transactions:
        Chronologically ordered transaction stream.
    engine:
        Leakage-safe feature engine maintaining customer history.

    Yields
    ------
    dict
        Feature values plus the supervised fraud target.
    """

    for transaction, features in engine.compute(transactions):
        yield {
            "transaction_id": str(
                features.transaction_id
            ),
            "transaction_timestamp": (
                features.transaction_timestamp
            ),
            "amount": features.amount,
            "merchant_risk_score": (
                features.merchant_risk_score
            ),
            "is_home_country": (
                features.is_home_country
            ),
            "device_age_hours": (
                features.device_age_hours
            ),
            "transactions_before": (
                features.transactions_before
            ),
            "avg_amount_before": (
                features.avg_amount_before
            ),
            "amount_vs_avg": (
                features.amount_vs_avg
            ),
            "seconds_since_previous_tx": (
                features.seconds_since_previous_tx
            ),
            "tx_count_10m": (
                features.tx_count_10m
            ),
            "tx_count_1h": (
                features.tx_count_1h
            ),
            "tx_count_24h": (
                features.tx_count_24h
            ),
            "merchant_category_seen_before": (
                features.merchant_category_seen_before
            ),
            "is_fraud": bool(
                transaction.is_fraud
            ),
        }


def build_feature_dataset(
    output_path: Path = OUTPUT_PATH,
    max_rows: int | None = None,
) -> None:
    """
    Build the offline Parquet feature dataset.

    The function processes transactions in chronological order and writes
    bounded Arrow tables to a ParquetWriter. Only one feature batch is held
    in memory at a time.

    Parameters
    ----------
    output_path:
        Destination Parquet file.
    max_rows:
        Optional maximum number of transactions to process. This is useful
        for controlled dry runs. When None, all transactions are processed.

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If output_path already exists.
    """

    if output_path.exists():
        raise FileExistsError(
            f"Feature dataset already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    references = load_reference_data()

    engine = FeatureEngine(
        references=references
    )

    transactions = stream_transactions(
        batch_size=TRANSACTION_BATCH_SIZE
    )

    rows = feature_rows(
        transactions=transactions,
        engine=engine,
    )

    writer: pq.ParquetWriter | None = None

    buffer: list[dict] = []
    processed = 0

    try:
        for row in rows:
            buffer.append(row)

            if len(buffer) >= TRANSACTION_BATCH_SIZE:
                table = pa.Table.from_pylist(
                    buffer,
                    schema=FEATURE_SCHEMA,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        FEATURE_SCHEMA,
                        compression="zstd",
                    )

                writer.write_table(table)

                processed += len(buffer)

                if (
                    processed % PROGRESS_INTERVAL
                    < TRANSACTION_BATCH_SIZE
                ):
                    print(
                        f"Processed {processed:,} transactions"
                    )

                buffer.clear()

            if (
                max_rows is not None
                and processed + len(buffer)
                >= max_rows
            ):
                break

        if buffer:
            remaining = buffer

            if max_rows is not None:
                remaining = remaining[
                    : max_rows - processed
                ]

            if remaining:
                table = pa.Table.from_pylist(
                    remaining,
                    schema=FEATURE_SCHEMA,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        FEATURE_SCHEMA,
                        compression="zstd",
                    )

                writer.write_table(table)
                processed += len(remaining)

        print(
            f"Feature dataset written: {processed:,} rows"
        )

    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    """
    Build the full Shepherd feature dataset.

    Returns
    -------
    None
    """

    build_feature_dataset()


if __name__ == "__main__":
    main()