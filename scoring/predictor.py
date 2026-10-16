
"""Load Shepherd's approved scoring artifacts and produce auditable decisions.

This module provides a reusable inference component for Shepherd's fraud
detection system. It loads the calibration-experiment XGBoost model, the
sigmoid calibrator, and the saved feature schema.

For each transaction, the scorer:
    1. Validates the feature inputs.
    2. Computes the XGBoost raw margin.
    3. Converts the margin into a calibrated fraud probability.
    4. Applies the frozen validation-selected operating threshold.
    5. Returns an auditable decision record.

A transaction exceeding the threshold is marked REVIEW. A transaction below
the threshold is marked NO_ALERT. These are alerting outcomes, not an
automatic payment approval or rejection.

The scorer does not connect to PostgreSQL, Kafka, or FastAPI. Keeping inference
independent of these systems makes the scoring logic easier to test and reuse.

Usage:
    from scoring.predictor import FraudScorer

    scorer = FraudScorer()
    result = scorer.score_transaction(
        transaction_id="example-transaction",
        features=feature_mapping,
    )
    print(result.to_dict())
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "artifacts/models/xgboost_calibration_base.json"
)
CALIBRATORS_PATH = (
    PROJECT_ROOT / "artifacts/models/calibrators.joblib"
)
FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "artifacts/models/xgboost_calibration_base_features.json"
)

MODEL_VERSION = "shepherd-xgb-sigmoid-v1"
CALIBRATION_METHOD = "sigmoid"

# Selected on the validation partition for a minimum precision target of 50%.
# This is the operating point for this specific model and calibrator.
OPERATING_THRESHOLD = 0.23

# The null value for this feature is expected on a customer's first transaction.
NULLABLE_FEATURE = "seconds_since_previous_tx"

FORBIDDEN_FEATURES = {
    "is_fraud",
    "transaction_id",
    "transaction_timestamp",
}


@dataclass(frozen=True)
class FraudDecision:
    """Represent an auditable transaction scoring result.

    Attributes:
        transaction_id: Identifier of the transaction being scored.
        transaction_timestamp_utc: Original transaction time in UTC, if
            provided.
        model_version: Version identifier assigned to the scoring model.
        calibration_method: Calibration method applied to the raw model score.
        fraud_probability: Sigmoid-calibrated fraud probability.
        threshold: Frozen operating threshold used for the decision.
        decision: REVIEW or NO_ALERT.
        decision_reason: Explanation of the threshold-based decision.
        alert_triggered: Whether the transaction crossed the review threshold.
        raw_model_margin: XGBoost output before probability calibration.
        scored_at_utc: Time at which the scoring result was created.
    """

    transaction_id: str
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

    def to_dict(self) -> dict[str, Any]:
        """Convert the scoring result into a serialization-friendly mapping.

        Returns:
            A dictionary suitable for JSON serialization or a downstream
            persistence/event-publishing layer.
        """
        return asdict(self)


class FraudScorer:
    """Score individual transactions using Shepherd's calibrated model.

    Model artifacts are loaded once when the scorer is constructed. A service
    can therefore initialize one instance at startup and reuse it for multiple
    requests instead of reloading the model for every transaction.

    Only trusted model and calibrator artifacts should be loaded because
    joblib artifacts use Python serialization.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        calibrators_path: str | Path = CALIBRATORS_PATH,
        feature_list_path: str | Path = FEATURE_LIST_PATH,
        model_version: str = MODEL_VERSION,
        threshold: float = OPERATING_THRESHOLD,
        calibration_method: str = CALIBRATION_METHOD,
    ) -> None:
        """Load and validate the scoring artifacts.

        Args:
            model_path: Path to the saved XGBoost model.
            calibrators_path: Path to the saved calibration objects.
            feature_list_path: Path to the saved ordered feature list.
            model_version: Version identifier included in every result.
            threshold: Probability threshold for triggering a review alert.
            calibration_method: Name of the calibrator to load.

        Raises:
            FileNotFoundError:
                If a required artifact is missing.
            ValueError:
                If the threshold, feature schema, or calibrator is invalid.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "The operating threshold must be between zero and one."
            )

        if not model_version.strip():
            raise ValueError("The model version cannot be empty.")

        self.model_path = Path(model_path)
        self.calibrators_path = Path(calibrators_path)
        self.feature_list_path = Path(feature_list_path)

        self.model_version = model_version
        self.threshold = float(threshold)
        self.calibration_method = calibration_method

        for artifact_path in (
            self.model_path,
            self.calibrators_path,
            self.feature_list_path,
        ):
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Required scoring artifact not found: {artifact_path}"
                )

        with self.feature_list_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            feature_columns = json.load(file)

        if (
            not isinstance(feature_columns, list)
            or not feature_columns
            or not all(
                isinstance(column, str)
                for column in feature_columns
            )
        ):
            raise ValueError(
                "The saved feature schema must be a non-empty list "
                "of feature names."
            )

        if len(feature_columns) != len(set(feature_columns)):
            raise ValueError(
                "The saved feature schema contains duplicate names."
            )

        forbidden = set(feature_columns) & FORBIDDEN_FEATURES

        if forbidden:
            raise ValueError(
                f"Forbidden target or identifier fields in feature schema: "
                f"{sorted(forbidden)}"
            )

        self.feature_columns = feature_columns

        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))

        model_feature_names = self.model.feature_names

        if (
            model_feature_names is not None
            and model_feature_names != self.feature_columns
        ):
            raise ValueError(
                "The saved feature list does not match the feature order "
                "stored in the XGBoost model."
            )

        calibrators = joblib.load(self.calibrators_path)

        if not isinstance(calibrators, dict):
            raise ValueError(
                "The calibrator artifact must contain a dictionary."
            )

        if calibration_method not in calibrators:
            raise ValueError(
                f"Calibrator '{calibration_method}' was not found. "
                f"Available calibrators: {sorted(calibrators)}"
            )

        self.calibrator = calibrators[calibration_method]

        if not callable(
            getattr(self.calibrator, "predict_proba", None)
        ):
            raise ValueError(
                f"The '{calibration_method}' calibrator does not expose "
                "a predict_proba method."
            )

    @staticmethod
    def _format_utc_timestamp(
        value: datetime,
        field_name: str,
    ) -> str:
        """Validate a timezone-aware datetime and convert it to UTC.

        Args:
            value: Datetime supplied by the caller.
            field_name: Name used in error messages.

        Returns:
            An ISO-8601 UTC timestamp.

        Raises:
            TypeError:
                If the supplied value is not a datetime.
            ValueError:
                If the datetime has no usable timezone information.
        """
        if not isinstance(value, datetime):
            raise TypeError(
                f"{field_name} must be a datetime object."
            )

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"{field_name} must be timezone-aware."
            )

        return value.astimezone(timezone.utc).isoformat()

    def _prepare_feature_vector(
        self,
        features: Mapping[str, Any],
    ) -> np.ndarray:
        """Validate feature values and create the ordered model input.

        Only the feature names saved with the model are used. Missing values
        are rejected except for seconds_since_previous_tx, whose null value
        is represented as zero, matching the existing training pipeline.

        Args:
            features: Mapping from feature name to numeric value.

        Returns:
            A float32 feature vector with shape (1, number_of_features).

        Raises:
            TypeError:
                If features is not a mapping or a value is not numeric.
            ValueError:
                If required features are missing, forbidden fields are
                present, or values are invalid.
        """
        if not isinstance(features, Mapping):
            raise TypeError(
                "features must be a mapping of feature names to values."
            )

        supplied_names = set(features)

        forbidden = supplied_names & FORBIDDEN_FEATURES

        if forbidden:
            raise ValueError(
                f"Target or identifier fields cannot be passed as model "
                f"features: {sorted(forbidden)}"
            )

        missing = [
            column
            for column in self.feature_columns
            if column not in features
        ]

        if missing:
            raise ValueError(
                f"Required model features are missing: {missing}"
            )

        values: list[float] = []

        for column in self.feature_columns:
            value = features[column]

            if value is None:
                if column == NULLABLE_FEATURE:
                    numeric_value = 0.0
                else:
                    raise ValueError(
                        f"Feature '{column}' cannot be null."
                    )
            else:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"Feature '{column}' must be numeric."
                    ) from exc

            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Feature '{column}' must be finite; "
                    f"received {numeric_value!r}."
                )

            values.append(numeric_value)

        vector = np.asarray(
            values,
            dtype=np.float32,
        ).reshape(1, -1)

        if not np.isfinite(vector).all():
            raise ValueError(
                "Feature values are not finite after float32 conversion."
            )

        return vector

    def score_transaction(
        self,
        transaction_id: str,
        features: Mapping[str, Any],
        transaction_timestamp: datetime | None = None,
        scored_at: datetime | None = None,
    ) -> FraudDecision:
        """Score one transaction and return its auditable decision record.

        Args:
            transaction_id: Non-empty identifier for the transaction.
            features: Mapping containing the model's required feature values.
            transaction_timestamp: Optional timezone-aware time at which the
                original transaction occurred.
            scored_at: Optional timezone-aware scoring time, useful for
                deterministic tests. Defaults to the current UTC time.

        Returns:
            A FraudDecision containing the probability, threshold, decision,
            model version, score details, and timestamps.

        Raises:
            ValueError:
                If the transaction identifier or feature values are invalid.
            TypeError:
                If timestamp arguments are not datetime objects.
        """
        if not isinstance(transaction_id, str) or not transaction_id.strip():
            raise ValueError(
                "transaction_id must be a non-empty string."
            )

        vector = self._prepare_feature_vector(features)

        matrix = xgb.DMatrix(
            vector,
            feature_names=self.feature_columns,
        )

        raw_margin = float(
            self.model.predict(
                matrix,
                output_margin=True,
            )[0]
        )

        if not math.isfinite(raw_margin):
            raise ValueError(
                "The XGBoost model produced a non-finite raw margin."
            )

        calibrated_probability = float(
            self.calibrator.predict_proba(
                np.asarray(
                    [[raw_margin]],
                    dtype=np.float64,
                )
            )[0, 1]
        )

        if (
            not math.isfinite(calibrated_probability)
            or not 0.0 <= calibrated_probability <= 1.0
        ):
            raise ValueError(
                "The calibrator produced an invalid probability."
            )

        alert_triggered = (
            calibrated_probability >= self.threshold
        )

        if alert_triggered:
            decision = "REVIEW"
            decision_reason = (
                "calibrated_probability_at_or_above_threshold"
            )
        else:
            decision = "NO_ALERT"
            decision_reason = (
                "calibrated_probability_below_threshold"
            )

        if transaction_timestamp is None:
            transaction_timestamp_utc = None
        else:
            transaction_timestamp_utc = (
                self._format_utc_timestamp(
                    transaction_timestamp,
                    "transaction_timestamp",
                )
            )

        if scored_at is None:
            scored_at = datetime.now(timezone.utc)

        scored_at_utc = self._format_utc_timestamp(
            scored_at,
            "scored_at",
        )

        return FraudDecision(
            transaction_id=transaction_id,
            transaction_timestamp_utc=transaction_timestamp_utc,
            model_version=self.model_version,
            calibration_method=self.calibration_method,
            fraud_probability=calibrated_probability,
            threshold=self.threshold,
            decision=decision,
            decision_reason=decision_reason,
            alert_triggered=alert_triggered,
            raw_model_margin=raw_margin,
            scored_at_utc=scored_at_utc,
        )