"""Expose Shepherd's calibrated fraud scorer through a FastAPI service.

The service loads its model and sigmoid calibrator once during application
startup. Clients submit a transaction identifier, an optional transaction
timestamp, and the complete feature mapping required by the model.

Endpoints:
    GET  /health   Check whether the scoring service is ready.
    POST /v1/score Score one transaction and return an auditable decision.

The response includes the calibrated fraud probability, decision threshold,
decision, model version, raw XGBoost margin, and timestamps.

This initial API is intended for local development. It does not implement
authentication, authorization, rate limiting, or durable prediction storage.

Usage:
    python -m uvicorn scoring.api:app --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from scoring.predictor import FraudScorer


class ScoreRequest(BaseModel):
    """Represent a transaction-scoring API request.

    Attributes:
        transaction_id: Non-empty identifier for the transaction.
        transaction_timestamp: Optional timezone-aware transaction timestamp.
        features: Mapping containing all the model's required features.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(
        min_length=1,
        description="Unique transaction identifier.",
    )

    transaction_timestamp: datetime | None = Field(
        default=None,
        description="Transaction time, preferably expressed in UTC.",
    )

    features: dict[str, float | None] = Field(
        description=(
            "Complete model feature mapping. "
            "The fraud target must not be included."
        )
    )


class ScoreResponse(BaseModel):
    """Represent the auditable result returned by the scoring endpoint.

    Attributes:
        transaction_id: Transaction identifier supplied by the client.
        transaction_timestamp_utc: Transaction time converted to UTC.
        model_version: Version of the model used for inference.
        calibration_method: Probability calibration method.
        fraud_probability: Calibrated fraud probability between zero and one.
        threshold: Frozen alert threshold.
        decision: REVIEW or NO_ALERT.
        decision_reason: Machine-readable reason for the decision.
        alert_triggered: Whether the threshold was met.
        raw_model_margin: XGBoost raw margin before sigmoid calibration.
        scored_at_utc: Timestamp at which the scoring decision was created.
    """

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
    """Represent the scoring service's readiness status.

    Attributes:
        status: Readiness status of the API.
        model_version: Loaded model version.
    """

    status: Literal["ready"]
    model_version: str


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load scoring artifacts once and manage their application lifecycle.

    Args:
        application: FastAPI application instance.

    Yields:
        Control to the running application after model initialization.

    Raises:
        FileNotFoundError:
            If a required scoring artifact is missing.
        ValueError:
            If the model or calibration artifacts are invalid.
    """
    application.state.scorer = FraudScorer()

    try:
        yield
    finally:
        application.state.scorer = None


app = FastAPI(
    title="Shepherd Fraud Scoring API",
    description=(
        "Score transactions using Shepherd's XGBoost model and "
        "sigmoid-calibrated fraud probabilities."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_scorer(request: Request) -> FraudScorer:
    """Return the initialized scorer for an incoming request.

    Args:
        request: Incoming FastAPI request.

    Returns:
        The scorer loaded during application startup.

    Raises:
        HTTPException:
            If the scoring component is not ready.
    """
    scorer = getattr(request.app.state, "scorer", None)

    if not isinstance(scorer, FraudScorer):
        raise HTTPException(
            status_code=503,
            detail="Fraud scoring service is not ready.",
        )

    return scorer


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check scoring service readiness",
)
def health(request: Request) -> HealthResponse:
    """Report whether model artifacts have been loaded successfully.

    Args:
        request: Incoming health-check request.

    Returns:
        Readiness status and loaded model version.

    Raises:
        HTTPException:
            If the scoring service has not initialized.
    """
    scorer = get_scorer(request)

    return HealthResponse(
        status="ready",
        model_version=scorer.model_version,
    )


@app.post(
    "/v1/score",
    response_model=ScoreResponse,
    summary="Score a transaction",
)
def score_transaction(
    payload: ScoreRequest,
    request: Request,
) -> ScoreResponse:
    """Score a single transaction and return its auditable decision.

    The API verifies that the feature mapping contains exactly the feature
    names expected by the loaded model. Invalid inputs receive an HTTP 422
    response instead of being silently ignored.

    Args:
        payload: Validated transaction-scoring request.
        request: Incoming FastAPI request.

    Returns:
        Calibrated fraud prediction and decision metadata.

    Raises:
        HTTPException:
            422 for invalid features or timestamps.
            503 if the scorer is not ready.
    """
    scorer = get_scorer(request)

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
            transaction_id=payload.transaction_id,
            transaction_timestamp=payload.transaction_timestamp,
            features=payload.features,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return ScoreResponse(**result.to_dict())