from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database.models.base import Base


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    model_version: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )

    training_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    auc_pr: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    recall_at_target_precision: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    artifact_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    approved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )