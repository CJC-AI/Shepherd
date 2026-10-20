"""Define the database model for auditable fraud predictions.

The fraud_predictions table stores each scoring result associated with an
existing transaction. Multiple predictions can reference the same transaction,
allowing Shepherd to retain prediction history across rescoring and model
versions.

The Boolean decision column represents whether an alert was triggered:
True means REVIEW; False means NO_ALERT.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.models.base import Base


class FraudPrediction(Base):
    """Represent one persisted fraud-scoring result.

    Attributes:
        prediction_id: Unique identifier for this prediction record.
        transaction_id: Existing transaction associated with the prediction.
        model_version: Version identifier for the model used.
        fraud_probability: Calibrated fraud probability between zero and one.
        decision: True if an alert was triggered; otherwise False.
        threshold_used: Probability threshold used for the decision.
        calibration_method: Calibration method applied to the model output.
        decision_reason: Machine-readable explanation for the decision.
        raw_model_margin: XGBoost output before probability calibration.
        scored_at: Time the prediction was generated.
        created_at: Time the prediction record was created in the database.
    """

    __tablename__ = "fraud_predictions"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Deliberately not unique: a transaction can have multiple predictions
    # as models are updated or a transaction is rescored.
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("transactions.transaction_id"),
        nullable=False,
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

    # True means REVIEW; False means NO_ALERT.
    # The existing database column name is preserved for compatibility.
    decision: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    threshold_used: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    # These fields may be NULL for historical predictions that predate
    # the new scoring audit contract.
    calibration_method: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    decision_reason: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    raw_model_margin: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    scored_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )