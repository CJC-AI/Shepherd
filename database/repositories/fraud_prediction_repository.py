"""Persist fraud predictions and verify database readiness.

This repository is the database boundary for Shepherd's scoring workflow.
It validates the scoring result, verifies that its transaction exists, and
writes an auditable prediction row to PostgreSQL.

Database errors are wrapped in a repository-specific exception so the API
can return a controlled response without exposing database internals.
"""

import math
import uuid
from datetime import datetime
from typing import Any, Mapping

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database.models.fraud_prediction import FraudPrediction
from database.models.transaction import Transaction
from database.session import SessionLocal


class TransactionNotFoundError(LookupError):
    """Raised when a prediction refers to no persisted transaction."""


class PredictionPersistenceError(RuntimeError):
    """Raised when PostgreSQL cannot be reached or a prediction cannot be saved."""


class FraudPredictionRepository:
    """Store and verify auditable fraud-prediction records."""

    def check_connection(self) -> None:
        """Verify that PostgreSQL is reachable.

        Raises:
            PredictionPersistenceError:
                If the connection or health-check query fails.
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
            value: String or UUID identifying a transaction.

        Returns:
            Parsed transaction UUID.

        Raises:
            ValueError:
                If the identifier is not a valid UUID.
        """
        try:
            return uuid.UUID(str(value))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                "transaction_id must be a valid UUID."
            ) from exc

    @staticmethod
    def _parse_scored_at(value: Any) -> datetime:
        """Parse a timezone-aware scoring timestamp.

        Args:
            value: ISO-8601 timestamp produced by the scoring component.

        Returns:
            Timezone-aware datetime.

        Raises:
            ValueError:
                If the timestamp is missing, malformed, or timezone-naive.
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

        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise ValueError(
                "scored_at_utc must include timezone information."
            )

        return timestamp

    @staticmethod
    def _validate_score(
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate the fields required to persist a scoring result.

        Args:
            record: Dictionary produced by FraudDecision.to_dict().

        Returns:
            A validated and normalized prediction record.

        Raises:
            ValueError:
                If a required field or scoring invariant is invalid.
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

        probability = float(record["fraud_probability"])
        threshold = float(record["threshold"])
        raw_margin = float(record["raw_model_margin"])

        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                "fraud_probability must be finite and between zero and one."
            )

        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be finite and between zero and one."
            )

        if not math.isfinite(raw_margin):
            raise ValueError(
                "raw_model_margin must be finite."
            )

        if not isinstance(record["model_version"], str) or not record[
            "model_version"
        ].strip():
            raise ValueError("model_version must not be empty.")

        if not isinstance(record["calibration_method"], str) or not record[
            "calibration_method"
        ].strip():
            raise ValueError("calibration_method must not be empty.")

        if not isinstance(record["decision_reason"], str) or not record[
            "decision_reason"
        ].strip():
            raise ValueError("decision_reason must not be empty.")

        if record["decision"] not in {"REVIEW", "NO_ALERT"}:
            raise ValueError(
                "decision must be REVIEW or NO_ALERT."
            )

        alert_triggered = record["alert_triggered"]

        if not isinstance(alert_triggered, bool):
            raise ValueError("alert_triggered must be a Boolean.")

        expected_alert = probability >= threshold

        if alert_triggered != expected_alert:
            raise ValueError(
                "alert_triggered conflicts with the probability and threshold."
            )

        expected_decision = "REVIEW" if expected_alert else "NO_ALERT"

        if record["decision"] != expected_decision:
            raise ValueError(
                "decision conflicts with the probability and threshold."
            )

        if not isinstance(record["decision_reason"], str):
            raise ValueError("decision_reason must be a string.")

        scored_at = FraudPredictionRepository._parse_scored_at(
            record["scored_at_utc"]
        )

        return {
            "transaction_id": transaction_id,
            "model_version": record["model_version"],
            "calibration_method": record["calibration_method"],
            "fraud_probability": probability,
            "threshold_used": threshold,
            "decision": alert_triggered,
            "decision_reason": record["decision_reason"],
            "raw_model_margin": raw_margin,
            "scored_at": scored_at,
        }

    def save_prediction(
        self,
        record: Mapping[str, Any],
    ) -> uuid.UUID:
        """Persist a prediction for an existing transaction.

        Each successful invocation creates a new prediction record. Multiple
        records may reference the same transaction, preserving model-version
        and rescoring history.

        Args:
            record: Auditable result produced by the scoring component.

        Returns:
            The UUID of the newly persisted prediction.

        Raises:
            ValueError:
                If the record contains invalid values.
            TransactionNotFoundError:
                If the transaction does not exist in PostgreSQL.
            PredictionPersistenceError:
                If a database operation fails.
        """
        validated = self._validate_score(record)

        try:
            with SessionLocal() as session:
                transaction = session.get(
                    Transaction,
                    validated["transaction_id"],
                )

                if transaction is None:
                    raise TransactionNotFoundError(
                        "The referenced transaction does not exist."
                    )

                prediction = FraudPrediction(
                    transaction_id=validated["transaction_id"],
                    model_version=validated["model_version"],
                    calibration_method=validated["calibration_method"],
                    fraud_probability=validated["fraud_probability"],
                    threshold_used=validated["threshold_used"],
                    decision=validated["decision"],
                    decision_reason=validated["decision_reason"],
                    raw_model_margin=validated["raw_model_margin"],
                    scored_at=validated["scored_at"],
                )

                session.add(prediction)
                session.commit()
                session.refresh(prediction)

                return prediction.prediction_id

        except TransactionNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise PredictionPersistenceError(
                "PostgreSQL could not persist the prediction."
            ) from exc