
"""Define the database model for auditable fraud predictions.

A transaction can have multiple prediction records across model versions and
intentional rescoring attempts. Idempotency keys prevent accidental duplicate
records for the same request, while request fingerprints detect key reuse with
different inputs.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Index, String
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.models.base import Base


class FraudPrediction(Base):
    """Represent one persisted fraud-scoring result.

    Attributes:
        prediction_id: Unique identifier for this prediction record.
        transaction_id: Transaction associated with this prediction.
        model_version: Version of the model used for scoring.
        fraud_probability: Calibrated probability of fraud.
        decision: True for REVIEW and False for NO_ALERT.
        threshold_used: Operating threshold applied to the probability.
        calibration_method: Method used to calibrate the model score.
        decision_reason: Machine-readable explanation for the decision.
        raw_model_margin: XGBoost output before probability calibration.
        scored_at: Timestamp at which the scoring result was generated.
        idempotency_key: Client-supplied request key, when provided.
        request_fingerprint: SHA-256 fingerprint of the normalized request.
        created_at: Database-record creation timestamp.
    """

    __tablename__ = "fraud_predictions"

    __table_args__ = (
        Index(
            "uq_fraud_predictions_idempotency_key",
            "idempotency_key",
            unique=True,
        ),
    )

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

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
    decision: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    threshold_used: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

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

    # Nullable to preserve historical rows created before idempotency existed.
    idempotency_key: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    request_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
