
"""Integration tests for Shepherd's fraud-prediction repository.

These tests use the real development PostgreSQL database.

They verify:
    - Database connectivity.
    - Persistence of prediction audit fields.
    - Multiple prediction history records per transaction.
    - Rejection of predictions for missing transactions.
    - Decision and probability validation.
    - Idempotent replay of a previously saved result.
    - Rejection of idempotency-key reuse with a different fingerprint.

Only test-created prediction records are deleted during cleanup.

Usage:
    python -m pytest tests/test_prediction_repository.py -v
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from sqlalchemy import delete, select

from database.models.fraud_prediction import FraudPrediction
from database.models.transaction import Transaction
from database.repositories.fraud_prediction_repository import (
    FraudPredictionRepository,
    IdempotencyConflictError,
    TransactionNotFoundError,
)
from database.session import SessionLocal


@pytest.fixture(scope="module")
def repository() -> FraudPredictionRepository:
    """Provide one repository instance for this test module."""
    return FraudPredictionRepository()


@pytest.fixture(scope="module")
def existing_transaction_id() -> str:
    """Return an existing transaction UUID from the development database."""
    with SessionLocal() as session:
        transaction_id = session.scalar(
            select(Transaction.transaction_id).limit(1)
        )

    assert transaction_id is not None, (
        "The development database must contain at least one transaction."
    )

    return str(transaction_id)


def make_prediction_record(
    transaction_id: str,
    *,
    model_version: str = "repository-test-v1",
    fraud_probability: float = 0.40,
    threshold: float = 0.23,
) -> dict[str, Any]:
    """Build a valid prediction record for repository testing."""
    alert_triggered = fraud_probability >= threshold

    return {
        "transaction_id": transaction_id,
        "transaction_timestamp_utc": None,
        "model_version": model_version,
        "calibration_method": "sigmoid",
        "fraud_probability": fraud_probability,
        "threshold": threshold,
        "decision": "REVIEW" if alert_triggered else "NO_ALERT",
        "decision_reason": (
            "calibrated_probability_at_or_above_threshold"
            if alert_triggered
            else "calibrated_probability_below_threshold"
        ),
        "alert_triggered": alert_triggered,
        "raw_model_margin": 0.75,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def delete_test_predictions(
    prediction_ids: list[uuid.UUID],
) -> None:
    """Delete only prediction rows created by the current test."""
    if not prediction_ids:
        return

    with SessionLocal() as session:
        session.execute(
            delete(FraudPrediction).where(
                FraudPrediction.prediction_id.in_(prediction_ids)
            )
        )
        session.commit()


def test_database_connection(
    repository: FraudPredictionRepository,
) -> None:
    """Verify that the repository can reach PostgreSQL."""
    repository.check_connection()


def test_prediction_audit_fields_are_persisted(
    repository: FraudPredictionRepository,
    existing_transaction_id: str,
) -> None:
    """Verify that prediction audit fields survive a database round trip."""
    record = make_prediction_record(
        existing_transaction_id,
        model_version="repository-audit-test-v1",
        fraud_probability=0.40,
        threshold=0.23,
    )

    prediction_id = None

    try:
        prediction_id = repository.save_prediction(record)

        with SessionLocal() as session:
            saved = session.get(FraudPrediction, prediction_id)

            assert saved is not None
            assert str(saved.transaction_id) == existing_transaction_id
            assert saved.model_version == "repository-audit-test-v1"
            assert saved.calibration_method == "sigmoid"
            assert saved.fraud_probability == pytest.approx(0.40)
            assert saved.threshold_used == pytest.approx(0.23)
            assert saved.decision is True
            assert (
                saved.decision_reason
                == "calibrated_probability_at_or_above_threshold"
            )
            assert saved.raw_model_margin == pytest.approx(0.75)
            assert saved.scored_at is not None
            assert saved.scored_at.tzinfo is not None
            assert saved.created_at is not None

    finally:
        if prediction_id is not None:
            delete_test_predictions([prediction_id])


def test_multiple_predictions_for_one_transaction_are_preserved(
    repository: FraudPredictionRepository,
    existing_transaction_id: str,
) -> None:
    """Verify that separate rescoring operations create separate records."""
    records = [
        make_prediction_record(
            existing_transaction_id,
            model_version="repository-history-test-v1",
            fraud_probability=0.20,
        ),
        make_prediction_record(
            existing_transaction_id,
            model_version="repository-history-test-v2",
            fraud_probability=0.70,
        ),
    ]

    prediction_ids: list[uuid.UUID] = []

    try:
        for record in records:
            prediction_ids.append(
                repository.save_prediction(record)
            )

        assert len(set(prediction_ids)) == 2

        with SessionLocal() as session:
            saved_rows = session.scalars(
                select(FraudPrediction).where(
                    FraudPrediction.prediction_id.in_(prediction_ids)
                )
            ).all()

            assert len(saved_rows) == 2

            saved_by_version = {
                row.model_version: row
                for row in saved_rows
            }

            first = saved_by_version["repository-history-test-v1"]
            second = saved_by_version["repository-history-test-v2"]

            assert first.decision is False
            assert second.decision is True
            assert first.fraud_probability == pytest.approx(0.20)
            assert second.fraud_probability == pytest.approx(0.70)
            assert first.transaction_id == second.transaction_id

    finally:
        delete_test_predictions(prediction_ids)


def test_nonexistent_transaction_is_rejected(
    repository: FraudPredictionRepository,
) -> None:
    """Verify that nonexistent transaction references cannot be persisted."""
    record = make_prediction_record(str(uuid.uuid4()))

    with pytest.raises(TransactionNotFoundError):
        repository.save_prediction(record)


def test_inconsistent_alert_decision_is_rejected(
    repository: FraudPredictionRepository,
    existing_transaction_id: str,
) -> None:
    """Verify that inconsistent probability and decision values are rejected."""
    record = make_prediction_record(
        existing_transaction_id,
        fraud_probability=0.95,
        threshold=0.23,
    )

    record["decision"] = "NO_ALERT"
    record["alert_triggered"] = False
    record["decision_reason"] = (
        "calibrated_probability_below_threshold"
    )

    with pytest.raises(
        ValueError,
        match="alert_triggered conflicts with the probability and threshold",
    ):
        repository.save_prediction(record)


def test_non_finite_probability_is_rejected(
    repository: FraudPredictionRepository,
    existing_transaction_id: str,
) -> None:
    """Verify that NaN fraud probabilities cannot be persisted."""
    record = make_prediction_record(
        existing_transaction_id,
        fraud_probability=float("nan"),
    )

    with pytest.raises(
        ValueError,
        match="fraud_probability must be finite",
    ):
        repository.save_prediction(record)


def test_non_finite_raw_margin_is_rejected(
    repository: FraudPredictionRepository,
    existing_transaction_id: str,
) -> None:
    """Verify that infinite raw margins cannot be persisted."""
    record = make_prediction_record(existing_transaction_id)
    record["raw_model_margin"] = float("inf")

    with pytest.raises(
        ValueError,
        match="raw_model_margin must be finite",
    ):
        repository.save_prediction(record)


def test_identical_request_returns_same_prediction(
    repository: FraudPredictionRepository,
    existing_transaction_id: str,
) -> None:
    """Verify that repeating an identical keyed request returns its original row."""
    key = f"repository-replay-{uuid.uuid4()}"
    fingerprint = "a" * 64

    record = make_prediction_record(
        existing_transaction_id,
        model_version="repository-idempotency-test-v1",
        fraud_probability=0.40,
    )

    first = None

    try:
        first = repository.save_prediction_result(
            record,
            idempotency_key=key,
            request_fingerprint=fingerprint,
        )

        second = repository.save_prediction_result(
            record,
            idempotency_key=key,
            request_fingerprint=fingerprint,
        )

        assert first.prediction_id == second.prediction_id
        assert first.idempotent_replay is False
        assert second.idempotent_replay is True
        assert first.to_dict() == second.to_dict()

    finally:
        if first is not None:
            delete_test_predictions([first.prediction_id])


def test_reusing_key_for_different_request_is_rejected(
    repository: FraudPredictionRepository,
    existing_transaction_id: str,
) -> None:
    """Verify that a key cannot be reused with another payload fingerprint."""
    key = f"repository-conflict-{uuid.uuid4()}"
    first_fingerprint = "a" * 64
    different_fingerprint = "b" * 64

    record = make_prediction_record(
        existing_transaction_id,
        model_version="repository-idempotency-conflict-test",
        fraud_probability=0.40,
    )

    first = None

    try:
        first = repository.save_prediction_result(
            record,
            idempotency_key=key,
            request_fingerprint=first_fingerprint,
        )

        with pytest.raises(
            IdempotencyConflictError,
            match="different request payload",
        ):
            repository.save_prediction_result(
                record,
                idempotency_key=key,
                request_fingerprint=different_fingerprint,
            )

    finally:
        if first is not None:
            delete_test_predictions([first.prediction_id])
