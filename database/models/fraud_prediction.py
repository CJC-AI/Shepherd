import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.models.base import Base


class FraudPrediction(Base):
    __tablename__ = "fraud_predictions"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("transactions.transaction_id"),
        nullable=False,
        unique=True,
        index=True,
    )

    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    fraud_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    decision: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    threshold_used: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )