"""Expose Shepherd's calibrated fraud scorer through FastAPI.

The application loads the model during startup and persists each successful
prediction to PostgreSQL before returning its response.

Endpoints:
    GET  /health   Check model and database readiness.
    POST /v1/score Score a transaction and persist its prediction.

This is a local-development service. Authentication, authorization, rate
limiting, and production deployment hardening are not implemented here.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from database.repositories.fraud_prediction_repository import (
    FraudPredictionRepository,
    PredictionPersistenceError,
    TransactionNotFoundError,
)
from scoring.predictor import FraudScorer


logger = logging.getLogger(__name__)


class ScoreRequest(BaseModel):
    """Represent a transaction-scoring request."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: UUID = Field(
        description="UUID of an existing transaction in PostgreSQL."
    )

    transaction_timestamp: datetime | None = Field(
        default=None,
        description="Optional timezone-aware transaction timestamp.",
    )

    features: dict[str, float | None] = Field(
        description="Complete feature mapping required by the model."
    )


class ScoreResponse(BaseModel):
    """Represent the persisted scoring result returned by the API."""

    prediction_id: UUID
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
    """Represent readiness of the scoring model and database."""

    status: Literal["ready"]
    model_version: str
    database_status: Literal["connected"]


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize the scoring component and prediction repository."""
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
        "Score transactions with Shepherd's sigmoid-calibrated XGBoost "
        "model and persist decisions to PostgreSQL."
    ),
    version="1.1.0",
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
) -> ScoreResponse:
    """Score a transaction, persist its decision, and return the audit record.

    A successful response means both inference and database persistence
    completed. The transaction must already exist in PostgreSQL.

    Args:
        payload: Validated request with transaction identifier and features.
        request: Incoming FastAPI request.

    Returns:
        The auditable prediction result, including its database prediction ID.

    Raises:
        HTTPException:
            404 if the transaction does not exist.
            422 if the feature schema or feature values are invalid.
            503 if PostgreSQL cannot persist the result.
    """
    scorer = get_scorer(request)
    repository = get_repository(request)

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
        result = scorer.score_transaction(
            transaction_id=str(payload.transaction_id),
            transaction_timestamp=payload.transaction_timestamp,
            features=payload.features,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    try:
        prediction_id = repository.save_prediction(
            result.to_dict()
        )
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

    return ScoreResponse(
        prediction_id=prediction_id,
        **result.to_dict(),
    )