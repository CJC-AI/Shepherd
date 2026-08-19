import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.models.base import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    email_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    signup_country: Mapped[str] = mapped_column(
        String(2),
        nullable=False,
    )

    kyc_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    risk_rating: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        CheckConstraint(
            "kyc_status IN ('pending', 'verified', 'rejected')",
            name="ck_customers_kyc_status",
        ),
        CheckConstraint(
            "risk_rating IS NULL OR risk_rating BETWEEN 1 AND 5",
            name="ck_customers_risk_rating",
        ),
    )