import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.models.base import Base


class Merchant(Base):
    __tablename__ = "merchants"

    merchant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    merchant_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    merchant_category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    merchant_country: Mapped[str] = mapped_column(
        String(2),
        nullable=False,
    )

    risk_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )