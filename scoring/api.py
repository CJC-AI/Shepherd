
"""Expose Shepherd's calibrated fraud scorer through FastAPI.

The model and calibrator load at application startup. Each scoring request
must include an Idempotency-Key header.

Repeated requests with the same key and equivalent payload return the
original persisted result. Reusing the key for different input returns HTTP
409 Conflict.

Endpoints:
    GET  /health
    POST /v1/score

This local-development service does not yet implement authentication,
authorization, rate limiting, or production deployment hardening.
"""

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from database.repositories.fraud_prediction_repository import (
    FraudPredictionRepository,
    IdempotencyConflictError,
    PredictionPersistenceError,
    TransactionNotFoundError,
)
from scoring.predictor import FraudScorer


logger = logging.getLogger(__name__)


class ScoreRequest(BaseModel):
    """Represent a validated transaction-scoring request."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(
        min_length=1,
        description="UUID of an existing transaction in PostgreSQL.",
    )

    transaction_timestamp: datetime | None = Field(
        default=None,
        description="Optional timezone-aware transaction timestamp.",
    )

    features: dict[str, float | None] = Field(
        description="Complete model feature mapping."
    )


class ScoreResponse(BaseModel):
    """Represent a persisted scoring decision."""

    prediction_id: str
    transaction_id: str
    transaction_timestamp_utc: str | None
    model_version: str
    calibration_method: str
    fraud_probability: float
    threshold: float
    decision: Literal["REVIEW", "NO_ALERT"]
    decision_reason: str
    alert_triggered: bool
    raw_model_margin: float
    scored_at_utc: str


class HealthResponse(BaseModel):
    """Represent scoring-model and database readiness."""

    status: Literal["ready"]
    model_version: str
    database_status: Literal["connected"]


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize service dependencies and release application references."""
    application.state.scorer = FraudScorer()
    application.state.prediction_repository = FraudPredictionRepository()

    try:
        yield
    finally:
        application.state.scorer = None
        application.state.prediction_repository = None


app = FastAPI(
    title="Shepherd Fraud Scoring API",
    description=(
        "Score transactions with Shepherd's calibrated XGBoost model, "
        "persist audit results, and support idempotent retries."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


def get_scorer(request: Request) -> FraudScorer:
    """Return the initialized scorer or raise a readiness error."""
    scorer = getattr(request.app.state, "scorer", None)

    if not isinstance(scorer, FraudScorer):
        raise HTTPException(
            status_code=503,
            detail="Fraud scoring service is not ready.",
        )

    return scorer


def get_repository(request: Request) -> FraudPredictionRepository:
    """Return the initialized prediction repository."""
    repository = getattr(
        request.app.state,
        "prediction_repository",
        None,
    )

    if not isinstance(repository, FraudPredictionRepository):
        raise HTTPException(
            status_code=503,
            detail="Prediction persistence is not ready.",
        )

    return repository


def fingerprint_request(payload: ScoreRequest) -> str:
    """Calculate a stable SHA-256 fingerprint of the scoring request.

    The transaction timestamp is normalized to UTC so equivalent timestamps
    expressed with different UTC offsets receive the same fingerprint.

    Feature keys are sorted before serialization so JSON key order does not
    affect the fingerprint.

    Args:
        payload: Validated request body.

    Returns:
        A 64-character lowercase SHA-256 digest.

    Raises:
        ValueError: If the timestamp or feature values cannot be normalized.
    """
    timestamp = payload.transaction_timestamp

    if timestamp is not None:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(
                "transaction_timestamp must be timezone-aware."
            )

        timestamp_utc = timestamp.astimezone(timezone.utc).isoformat()
    else:
        timestamp_utc = None

    canonical_payload = {
        "transaction_id": payload.transaction_id,
        "transaction_timestamp_utc": timestamp_utc,
        "features": {
            name: payload.features[name]
            for name in sorted(payload.features)
        },
    }

    canonical_json = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check model and database readiness",
)
def health(request: Request) -> HealthResponse:
    """Verify that the scorer is loaded and PostgreSQL is reachable."""
    scorer = get_scorer(request)
    repository = get_repository(request)

    try:
        repository.check_connection()
    except PredictionPersistenceError as exc:
        logger.exception("PostgreSQL readiness check failed.")

        raise HTTPException(
            status_code=503,
            detail="Prediction database is unavailable.",
        ) from exc

    return HealthResponse(
        status="ready",
        model_version=scorer.model_version,
        database_status="connected",
    )


@app.post(
    "/v1/score",
    response_model=ScoreResponse,
    summary="Score and persist a transaction",
)
def score_transaction(
    payload: ScoreRequest,
    request: Request,
    response: Response,
    idempotency_key: str = Header(
        ...,
        alias="Idempotency-Key",
        min_length=1,
        max_length=128,
        description=(
            "Unique client-generated key. Reuse it only when retrying "
            "the exact same scoring request."
        ),
    ),
) -> ScoreResponse:
    """Score, persist, and return an auditable transaction decision.

    Args:
        payload: Transaction identifier, timestamp, and feature mapping.
        request: Incoming request context.
        response: FastAPI response object for setting replay headers.
        idempotency_key: Client-generated unique request key.

    Returns:
        Persisted prediction record.

    Raises:
        HTTPException:
            404 if the transaction does not exist.
            409 if a key is reused with a different payload.
            422 for invalid input.
            503 if persistence or scoring readiness fails.
    """
    scorer = get_scorer(request)
    repository = get_repository(request)

    idempotency_key = idempotency_key.strip()

    if not idempotency_key:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key must not be empty.",
        )

    supplied_features = set(payload.features)
    expected_features = set(scorer.feature_columns)

    missing_features = sorted(
        expected_features - supplied_features
    )

    unexpected_features = sorted(
        supplied_features - expected_features
    )

    if missing_features or unexpected_features:
        detail: dict[str, Any] = {
            "message": "The submitted feature schema is invalid."
        }

        if missing_features:
            detail["missing_features"] = missing_features

        if unexpected_features:
            detail["unexpected_features"] = unexpected_features

        raise HTTPException(
            status_code=422,
            detail=detail,
        )

    try:
        request_fingerprint = fingerprint_request(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Fast path: return the original stored result without rescoring.
    try:
        existing = repository.find_idempotent_result(
            idempotency_key,
            request_fingerprint,
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    except PredictionPersistenceError as exc:
        logger.exception("Idempotency lookup failed.")

        raise HTTPException(
            status_code=503,
            detail="Prediction persistence is unavailable.",
        ) from exc

    if existing is not None:
        response.headers["Idempotency-Replayed"] = "true"
        return ScoreResponse(**existing.to_dict())

    # No previous result exists for this key, so perform inference.
    try:
        scoring_result = scorer.score_transaction(
            transaction_id=payload.transaction_id,
            transaction_timestamp=payload.transaction_timestamp,
            features=payload.features,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # The repository checks the key again during the write. This second check
    # is important when two identical requests arrive concurrently.
    try:
        saved_result = repository.save_prediction_result(
            scoring_result.to_dict(),
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    except TransactionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="The referenced transaction does not exist.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except PredictionPersistenceError as exc:
        logger.exception("Failed to persist fraud prediction.")

        raise HTTPException(
            status_code=503,
            detail="The prediction could not be persisted.",
        ) from exc

    response.headers["Idempotency-Replayed"] = (
        "true" if saved_result.idempotent_replay else "false"
    )

    return ScoreResponse(**saved_result.to_dict())
