"""
Define the database model for persisted bank transactions.

This module maps simulated transaction data to the PostgreSQL transactions
table and defines the foreign-key relationships to accounts, merchants, and
devices.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Numeric, String
from sqlalchemy.dialects.postgresql import INET, UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.models.base import Base


class Transaction(Base):
    """
    Represent a persisted bank transaction.

    A transaction links an account, merchant, and device and stores the
    monetary, temporal, geographic, and fraud-label attributes needed by the
    Shepherd fraud-detection pipeline.
    """

    __tablename__ = "transactions"

    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("accounts.account_id"),
        nullable=False,
        index=True,
    )

    merchant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("merchants.merchant_id"),
        nullable=False,
        index=True,
    )

    device_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("devices.device_id"),
        nullable=False,
        index=True,
    )

    amount: Mapped[float] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
    )

    transaction_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    transaction_country: Mapped[str] = mapped_column(
        String(2),
        nullable=False,
    )

    ip_address: Mapped[str] = mapped_column(
        INET,
        nullable=False,
    )

    is_fraud: Mapped[bool | None] = mapped_column(
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )