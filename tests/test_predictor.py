"""Automated tests for Shepherd's calibrated fraud scorer.

These tests exercise the real scoring artifacts and verify the behavior of
FraudScorer, including valid predictions, feature validation, timestamp
validation, reproducibility, and threshold-based decisions.

The model and calibrator artifacts must exist before these tests are run.

Usage:
    python -m pytest tests/test_predictor.py -v
"""

from datetime import datetime, timezone
import math

import pytest

from scoring.predictor import (
    FraudDecision,
    FraudScorer,
    OPERATING_THRESHOLD,
)


@pytest.fixture(scope="module")
def scorer() -> FraudScorer:
    """Load one scorer instance for reuse across this test module.

    Returns:
        A FraudScorer initialized with the real saved model and calibrator.
    """
    return FraudScorer()


@pytest.fixture
def valid_features(
    scorer: FraudScorer,
) -> dict[str, float]:
    """Build a complete, valid feature mapping for scoring tests.

    All required features receive a numeric value. Common transaction
    characteristics are then populated with representative values.

    Args:
        scorer: The initialized scorer whose saved schema defines
            the required feature names.

    Returns:
        A feature mapping containing every required model feature.
    """
    features = {
        column: 0.0
        for column in scorer.feature_columns
    }

    representative_values = {
        "amount": 100.0,
        "merchant_risk_score": 0.20,
        "is_home_country": 1.0,
        "device_age_hours": 24.0,
        "transactions_before": 10.0,
        "avg_amount_before": 100.0,
        "amount_vs_avg": 1.0,
        "seconds_since_previous_tx": 3600.0,
        "tx_count_10m": 0.0,
        "tx_count_1h": 1.0,
        "tx_count_24h": 3.0,
        "merchant_category_seen_before": 1.0,
    }

    for feature_name, value in representative_values.items():
        if feature_name in features:
            features[feature_name] = value

    return features


def test_valid_transaction_returns_auditable_decision(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that valid inputs produce a complete decision record."""
    transaction_time = datetime(
        2026, 8, 20, 12, 0, tzinfo=timezone.utc
    )

    scoring_time = datetime(
        2026, 10, 16, 12, 0, tzinfo=timezone.utc
    )

    result = scorer.score_transaction(
        transaction_id="test-transaction-001",
        features=valid_features,
        transaction_timestamp=transaction_time,
        scored_at=scoring_time,
    )

    assert isinstance(result, FraudDecision)
    assert result.transaction_id == "test-transaction-001"
    assert result.model_version
    assert result.calibration_method == "sigmoid"

    assert math.isfinite(result.fraud_probability)
    assert 0.0 <= result.fraud_probability <= 1.0

    assert result.threshold == OPERATING_THRESHOLD
    assert math.isfinite(result.raw_model_margin)

    assert result.transaction_timestamp_utc == (
        "2026-08-20T12:00:00+00:00"
    )

    assert result.scored_at_utc == (
        "2026-10-16T12:00:00+00:00"
    )

    assert result.decision == (
        "REVIEW"
        if result.fraud_probability >= result.threshold
        else "NO_ALERT"
    )

    assert result.alert_triggered == (
        result.fraud_probability >= result.threshold
    )

    assert isinstance(result.to_dict(), dict)


def test_missing_required_feature_is_rejected(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that omitting a required model feature raises an error."""
    valid_features.pop("amount")

    with pytest.raises(
        ValueError,
        match="Required model features are missing",
    ):
        scorer.score_transaction(
            transaction_id="test-missing-feature",
            features=valid_features,
        )


def test_target_leakage_is_rejected(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that the fraud label cannot be supplied as a feature."""
    valid_features["is_fraud"] = 1.0

    with pytest.raises(
        ValueError,
        match="Target or identifier fields cannot be passed",
    ):
        scorer.score_transaction(
            transaction_id="test-target-leakage",
            features=valid_features,
        )


def test_non_finite_feature_is_rejected(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that a NaN feature value cannot reach model inference."""
    valid_features["amount"] = float("nan")

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        scorer.score_transaction(
            transaction_id="test-nan-feature",
            features=valid_features,
        )


def test_infinite_feature_is_rejected(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that infinite feature values are rejected."""
    valid_features["amount"] = float("inf")

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        scorer.score_transaction(
            transaction_id="test-infinite-feature",
            features=valid_features,
        )


def test_non_numeric_feature_is_rejected(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that invalid feature types produce a clear error."""
    valid_features["amount"] = "not-a-number"

    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        scorer.score_transaction(
            transaction_id="test-invalid-type",
            features=valid_features,
        )


def test_first_transaction_null_is_supported(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that the expected previous-transaction null is accepted."""
    valid_features["seconds_since_previous_tx"] = None

    result = scorer.score_transaction(
        transaction_id="test-first-transaction",
        features=valid_features,
    )

    assert isinstance(result, FraudDecision)
    assert math.isfinite(result.fraud_probability)


def test_naive_transaction_timestamp_is_rejected(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that transaction timestamps must include timezone information."""
    naive_timestamp = datetime(2026, 8, 20, 12, 0)

    with pytest.raises(
        ValueError,
        match="must be timezone-aware",
    ):
        scorer.score_transaction(
            transaction_id="test-naive-timestamp",
            features=valid_features,
            transaction_timestamp=naive_timestamp,
        )


def test_naive_scoring_timestamp_is_rejected(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that scoring timestamps must include timezone information."""
    naive_timestamp = datetime(2026, 10, 16, 12, 0)

    with pytest.raises(
        ValueError,
        match="must be timezone-aware",
    ):
        scorer.score_transaction(
            transaction_id="test-naive-scoring-time",
            features=valid_features,
            scored_at=naive_timestamp,
        )


def test_empty_transaction_id_is_rejected(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify that a transaction cannot be scored without an identifier."""
    with pytest.raises(
        ValueError,
        match="transaction_id must be a non-empty string",
    ):
        scorer.score_transaction(
            transaction_id=" ",
            features=valid_features,
        )


def test_invalid_threshold_is_rejected() -> None:
    """Verify that a scoring threshold outside zero through one is invalid."""
    with pytest.raises(
        ValueError,
        match="operating threshold must be between zero and one",
    ):
        FraudScorer(threshold=1.01)


def test_identical_inputs_produce_identical_scores(
    scorer: FraudScorer,
    valid_features: dict[str, float],
) -> None:
    """Verify repeatable inference for identical inputs and artifacts."""
    fixed_time = datetime(
        2026, 10, 16, 12, 0, tzinfo=timezone.utc
    )

    first_result = scorer.score_transaction(
        transaction_id="test-repeatability",
        features=valid_features,
        scored_at=fixed_time,
    )

    second_result = scorer.score_transaction(
        transaction_id="test-repeatability",
        features=valid_features,
        scored_at=fixed_time,
    )

    assert first_result.fraud_probability == pytest.approx(
        second_result.fraud_probability,
        abs=1e-12,
    )

    assert first_result.raw_model_margin == pytest.approx(
        second_result.raw_model_margin,
        abs=1e-12,
    )

    assert first_result.decision == second_result.decision
    assert first_result.to_dict() == second_result.to_dict()