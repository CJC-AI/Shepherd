
"""Persist fraud predictions with request-level idempotency.

The repository validates prediction results, verifies transaction existence,
and stores auditable results in PostgreSQL.

An idempotency key is optional for internal callers. When supplied, it must
be paired with a SHA-256 request fingerprint. Repeating the same key and
fingerprint returns the original result. Reusing a key with a different
fingerprint raises IdempotencyConflictError.

Database errors are translated into repository-specific exceptions so callers
can return controlled HTTP responses without leaking database details.
"""

import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from database.models.fraud_prediction import FraudPrediction
from database.models.transaction import Transaction
from database.session import SessionLocal


class TransactionNotFoundError(LookupError):
    """Raised when a prediction references no existing transaction."""


class PredictionPersistenceError(RuntimeError):
    """Raised when PostgreSQL cannot complete a prediction operation."""


class IdempotencyConflictError(ValueError):
    """Raised when a key is reused with a different request fingerprint."""


@dataclass(frozen=True)
class PersistedPrediction:
    """Represent a prediction result read from or written to PostgreSQL.

    Attributes:
        prediction_id: Database ID of the saved prediction.
        transaction_id: UUID of the scored transaction.
        transaction_timestamp_utc: Actual transaction timestamp from PostgreSQL.
        model_version: Model version used for scoring.
        calibration_method: Method used to calibrate the model output.
        fraud_probability: Stored calibrated fraud probability.
        threshold: Stored operating threshold.
        decision: REVIEW or NO_ALERT.
        decision_reason: Machine-readable decision reason.
        alert_triggered: Whether the decision crossed the threshold.
        raw_model_margin: Raw XGBoost margin before calibration.
        scored_at_utc: Original persisted scoring timestamp.
        idempotent_replay: Whether this response represents a prior result.
    """

    prediction_id: uuid.UUID
    transaction_id: uuid.UUID
    transaction_timestamp_utc: str | None
    model_version: str
    calibration_method: str
    fraud_probability: float
    threshold: float
    decision: str
    decision_reason: str
    alert_triggered: bool
    raw_model_margin: float
    scored_at_utc: str
    idempotent_replay: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the API response fields, excluding internal replay status.

        The API indicates a replay using the Idempotency-Replayed response
        header. Keeping this flag out of the body means the replayed body can
        exactly match the original successful response.
        """
        return {
            "prediction_id": str(self.prediction_id),
            "transaction_id": str(self.transaction_id),
            "transaction_timestamp_utc": self.transaction_timestamp_utc,
            "model_version": self.model_version,
            "calibration_method": self.calibration_method,
            "fraud_probability": self.fraud_probability,
            "threshold": self.threshold,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "alert_triggered": self.alert_triggered,
            "raw_model_margin": self.raw_model_margin,
            "scored_at_utc": self.scored_at_utc,
        }


class FraudPredictionRepository:
    """Store, retrieve, and validate auditable fraud-prediction records."""

    def check_connection(self) -> None:
        """Verify that PostgreSQL is reachable.

        Raises:
            PredictionPersistenceError: If the connection check fails.
        """
        try:
            with SessionLocal() as session:
                session.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise PredictionPersistenceError(
                "Could not verify the PostgreSQL connection."
            ) from exc

    @staticmethod
    def _parse_transaction_id(value: Any) -> uuid.UUID:
        """Parse a transaction identifier as a UUID.

        Args:
            value: UUID or UUID-formatted string.

        Returns:
            Parsed UUID.

        Raises:
            ValueError: If the identifier is invalid.
        """
        try:
            return uuid.UUID(str(value))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                "transaction_id must be a valid UUID."
            ) from exc

    @staticmethod
    def _parse_scored_at(value: Any) -> datetime:
        """Parse a timezone-aware ISO-8601 scoring timestamp.

        Args:
            value: Timestamp string.

        Returns:
            Timezone-aware datetime.

        Raises:
            ValueError: If the timestamp is missing or invalid.
        """
        if not isinstance(value, str):
            raise ValueError(
                "scored_at_utc must be an ISO-8601 timestamp string."
            )

        try:
            timestamp = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "scored_at_utc is not a valid ISO-8601 timestamp."
            ) from exc

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(
                "scored_at_utc must include timezone information."
            )

        return timestamp.astimezone(timezone.utc)

    @staticmethod
    def _validate_idempotency(
        idempotency_key: str | None,
        request_fingerprint: str | None,
    ) -> tuple[str | None, str | None]:
        """Validate and normalize the optional idempotency metadata.

        Args:
            idempotency_key: Client-supplied request key.
            request_fingerprint: SHA-256 fingerprint of the request payload.

        Returns:
            Normalized key and fingerprint, or (None, None).

        Raises:
            ValueError: If the fields are malformed or only one is supplied.
        """
        if idempotency_key is None and request_fingerprint is None:
            return None, None

        if idempotency_key is None or request_fingerprint is None:
            raise ValueError(
                "idempotency_key and request_fingerprint must be supplied "
                "together."
            )

        if not isinstance(idempotency_key, str):
            raise ValueError("idempotency_key must be a string.")

        key = idempotency_key.strip()

        if not key or len(key) > 128:
            raise ValueError(
                "idempotency_key must contain 1 to 128 characters."
            )

        if not isinstance(request_fingerprint, str):
            raise ValueError(
                "request_fingerprint must be a string."
            )

        fingerprint = request_fingerprint.lower()

        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError(
                "request_fingerprint must be a 64-character SHA-256 hex digest."
            )

        return key, fingerprint

    @staticmethod
    def _validate_score(
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate and normalize the scorer's audit record.

        Args:
            record: Dictionary returned by FraudDecision.to_dict().

        Returns:
            Normalized values for the database write.

        Raises:
            ValueError: If a required field or decision invariant is invalid.
        """
        required = {
            "transaction_id",
            "model_version",
            "calibration_method",
            "fraud_probability",
            "threshold",
            "decision",
            "decision_reason",
            "alert_triggered",
            "raw_model_margin",
            "scored_at_utc",
        }

        missing = required - set(record)

        if missing:
            raise ValueError(
                f"Prediction record is missing fields: {sorted(missing)}"
            )

        transaction_id = FraudPredictionRepository._parse_transaction_id(
            record["transaction_id"]
        )

        try:
            probability = float(record["fraud_probability"])
            threshold = float(record["threshold"])
            raw_margin = float(record["raw_model_margin"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Probability, threshold, and model margin must be numeric."
            ) from exc

        if (
            not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError(
                "fraud_probability must be finite and between zero and one."
            )

        if (
            not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError(
                "threshold must be finite and between zero and one."
            )

        if not math.isfinite(raw_margin):
            raise ValueError("raw_model_margin must be finite.")

        model_version = record["model_version"]
        calibration_method = record["calibration_method"]
        decision_reason = record["decision_reason"]
        decision = record["decision"]
        alert_triggered = record["alert_triggered"]

        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must not be empty.")

        if len(model_version) > 50:
            raise ValueError("model_version cannot exceed 50 characters.")

        if (
            not isinstance(calibration_method, str)
            or not calibration_method.strip()
        ):
            raise ValueError("calibration_method must not be empty.")

        if len(calibration_method) > 30:
            raise ValueError(
                "calibration_method cannot exceed 30 characters."
            )

        if not isinstance(decision_reason, str) or not decision_reason.strip():
            raise ValueError("decision_reason must not be empty.")

        if len(decision_reason) > 100:
            raise ValueError(
                "decision_reason cannot exceed 100 characters."
            )

        if decision not in {"REVIEW", "NO_ALERT"}:
            raise ValueError("decision must be REVIEW or NO_ALERT.")

        if not isinstance(alert_triggered, bool):
            raise ValueError("alert_triggered must be a Boolean.")

        expected_alert = probability >= threshold

        if alert_triggered != expected_alert:
            raise ValueError(
                "alert_triggered conflicts with the probability and threshold."
            )

        expected_decision = "REVIEW" if expected_alert else "NO_ALERT"

        if decision != expected_decision:
            raise ValueError(
                "decision conflicts with the probability and threshold."
            )

        return {
            "transaction_id": transaction_id,
            "model_version": model_version,
            "calibration_method": calibration_method,
            "fraud_probability": probability,
            "threshold_used": threshold,
            "decision": alert_triggered,
            "decision_reason": decision_reason,
            "raw_model_margin": raw_margin,
            "scored_at": FraudPredictionRepository._parse_scored_at(
                record["scored_at_utc"]
            ),
        }

    @staticmethod
    def _timestamp_to_iso(value: datetime | None) -> str | None:
        """Convert a database datetime to an ISO-8601 UTC string.

        Args:
            value: Database timestamp.

        Returns:
            UTC timestamp string or None if the value is absent.
        """
        if value is None:
            return None

        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc).isoformat()

    def _to_persisted_result(
        self,
        session: Any,
        prediction: FraudPrediction,
        *,
        idempotent_replay: bool,
    ) -> PersistedPrediction:
        """Convert a saved prediction and its transaction into a result.

        Args:
            session: Active SQLAlchemy session.
            prediction: Persisted prediction ORM instance.
            idempotent_replay: Whether this result represents a prior request.

        Returns:
            A detached, serialization-friendly prediction result.

        Raises:
            TransactionNotFoundError: If the referenced transaction is absent.
        """
        transaction = session.get(
            Transaction,
            prediction.transaction_id,
        )

        if transaction is None:
            raise TransactionNotFoundError(
                "The referenced transaction does not exist."
            )

        if prediction.scored_at is None:
            raise PredictionPersistenceError(
                "The stored prediction is missing its scoring timestamp."
            )

        if prediction.raw_model_margin is None:
            raise PredictionPersistenceError(
                "The stored prediction is missing its raw model margin."
            )

        if prediction.calibration_method is None:
            raise PredictionPersistenceError(
                "The stored prediction is missing its calibration method."
            )

        if prediction.decision_reason is None:
            raise PredictionPersistenceError(
                "The stored prediction is missing its decision reason."
            )

        return PersistedPrediction(
            prediction_id=prediction.prediction_id,
            transaction_id=prediction.transaction_id,
            transaction_timestamp_utc=self._timestamp_to_iso(
                transaction.transaction_timestamp
            ),
            model_version=prediction.model_version,
            calibration_method=prediction.calibration_method,
            fraud_probability=float(prediction.fraud_probability),
            threshold=float(prediction.threshold_used),
            decision="REVIEW" if prediction.decision else "NO_ALERT",
            decision_reason=prediction.decision_reason,
            alert_triggered=bool(prediction.decision),
            raw_model_margin=float(prediction.raw_model_margin),
            scored_at_utc=self._timestamp_to_iso(
                prediction.scored_at
            ) or "",
            idempotent_replay=idempotent_replay,
        )

    @staticmethod
    def _find_by_key(
        session: Any,
        idempotency_key: str,
    ) -> FraudPrediction | None:
        """Retrieve a prediction using its unique idempotency key.

        Args:
            session: Active SQLAlchemy session.
            idempotency_key: Normalized request key.

        Returns:
            Existing prediction or None.
        """
        return session.scalar(
            select(FraudPrediction).where(
                FraudPrediction.idempotency_key == idempotency_key
            )
        )

    def find_idempotent_result(
        self,
        idempotency_key: str,
        request_fingerprint: str,
    ) -> PersistedPrediction | None:
        """Return a prior result if the same request has already succeeded.

        Args:
            idempotency_key: Client-supplied request key.
            request_fingerprint: SHA-256 fingerprint of the request.

        Returns:
            The original stored result, marked internally as a replay, or
            None if the key has not yet been used.

        Raises:
            IdempotencyConflictError: If the key exists with another fingerprint.
            PredictionPersistenceError: If PostgreSQL cannot be queried.
        """
        key, fingerprint = self._validate_idempotency(
            idempotency_key,
            request_fingerprint,
        )

        assert key is not None
        assert fingerprint is not None

        try:
            with SessionLocal() as session:
                existing = self._find_by_key(session, key)

                if existing is None:
                    return None

                if existing.request_fingerprint != fingerprint:
                    raise IdempotencyConflictError(
                        "This Idempotency-Key has already been used with a "
                        "different request payload."
                    )

                return self._to_persisted_result(
                    session,
                    existing,
                    idempotent_replay=True,
                )
        except IdempotencyConflictError:
            raise
        except SQLAlchemyError as exc:
            raise PredictionPersistenceError(
                "PostgreSQL could not look up the idempotency key."
            ) from exc

    def save_prediction(
        self,
        record: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
    ) -> uuid.UUID:
        """Persist a prediction and return its database identifier.

        This compatibility method preserves the original repository API.
        For API responses that need to return the stored record on retries,
        use save_prediction_result().

        Args:
            record: Auditable scoring result.
            idempotency_key: Optional request-level idempotency key.
            request_fingerprint: Optional SHA-256 request fingerprint.

        Returns:
            UUID of the newly created or previously persisted prediction.
        """
        result = self.save_prediction_result(
            record,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
        )
        return result.prediction_id

    def save_prediction_result(
        self,
        record: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
    ) -> PersistedPrediction:
        """Persist a prediction or return the previous idempotent result.

        If simultaneous requests race to store the same key, the unique index
        ensures only one record wins. The losing request reads and returns the
        winner's result if its fingerprint matches.

        Args:
            record: Auditable scoring result.
            idempotency_key: Optional client-supplied request key.
            request_fingerprint: SHA-256 fingerprint when a key is supplied.

        Returns:
            Persisted prediction result, with replay status available
            internally to the API.

        Raises:
            ValueError: If prediction or idempotency data is invalid.
            TransactionNotFoundError: If the transaction does not exist.
            IdempotencyConflictError: If the key was reused for other inputs.
            PredictionPersistenceError: If persistence fails.
        """
        validated = self._validate_score(record)

        key, fingerprint = self._validate_idempotency(
            idempotency_key,
            request_fingerprint,
        )

        try:
            with SessionLocal() as session:
                if key is not None:
                    existing = self._find_by_key(session, key)

                    if existing is not None:
                        if existing.request_fingerprint != fingerprint:
                            raise IdempotencyConflictError(
                                "This Idempotency-Key has already been used "
                                "with a different request payload."
                            )

                        return self._to_persisted_result(
                            session,
                            existing,
                            idempotent_replay=True,
                        )

                transaction = session.get(
                    Transaction,
                    validated["transaction_id"],
                )

                if transaction is None:
                    raise TransactionNotFoundError(
                        "The referenced transaction does not exist."
                    )

                prediction = FraudPrediction(
                    **validated,
                    idempotency_key=key,
                    request_fingerprint=fingerprint,
                )

                session.add(prediction)
                session.flush()
                session.commit()
                session.refresh(prediction)

                return self._to_persisted_result(
                    session,
                    prediction,
                    idempotent_replay=False,
                )

        except (
            TransactionNotFoundError,
            IdempotencyConflictError,
        ):
            raise

        except IntegrityError as exc:
            # Another concurrent request may have committed this key after
            # the initial lookup. Resolve that race through the unique index.
            if key is not None and fingerprint is not None:
                try:
                    replay = self.find_idempotent_result(
                        key,
                        fingerprint,
                    )
                except IdempotencyConflictError:
                    raise
                except PredictionPersistenceError:
                    raise

                if replay is not None:
                    return replay

            raise PredictionPersistenceError(
                "PostgreSQL rejected the prediction write."
            ) from exc

        except SQLAlchemyError as exc:
            raise PredictionPersistenceError(
                "PostgreSQL could not persist the prediction."
            ) from exc
