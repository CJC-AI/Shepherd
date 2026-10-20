"""Exercise the running API using a real validation transaction.

This script loads one transaction and its corresponding features from the
feature dataset, then sends them to the locally running FastAPI service.

The fraud target is not loaded. The test verifies that the API returns a
prediction ID after successfully persisting a scoring result.

Usage:
    Start the API in one terminal:
        python -m uvicorn scoring.api:app --host 127.0.0.1 --port 8000

    Run this script in another terminal:
        python -m scoring.api_smoke_test
"""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATASET_PATH = (
    PROJECT_ROOT
    / "artifacts/features/transaction_features.parquet"
)

FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "artifacts/models/xgboost_calibration_base_features.json"
)

API_URL = "http://127.0.0.1:8000/v1/score"

VALIDATION_START_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85


def load_validation_request() -> dict[str, Any]:
    """Build a score request from a real validation feature row.

    Returns:
        JSON-compatible payload containing a real transaction UUID, its
        transaction timestamp, and its corresponding model features.
    """
    for path in (
        FEATURE_DATASET_PATH,
        FEATURE_LIST_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    with FEATURE_LIST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    required_columns = [
        "transaction_id",
        "transaction_timestamp",
        *feature_columns,
    ]

    df = pd.read_parquet(
        FEATURE_DATASET_PATH,
        columns=required_columns,
    )

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    if not df["transaction_timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Feature dataset is not chronologically ordered."
        )

    total_rows = len(df)

    validation_start = int(
        total_rows * VALIDATION_START_FRACTION
    )

    validation_end = int(
        total_rows * VALIDATION_END_FRACTION
    )

    validation_df = df.iloc[
        validation_start:validation_end
    ]

    if validation_df.empty:
        raise ValueError("Validation partition is empty.")

    row = validation_df.iloc[0]

    features: dict[str, float | None] = {}

    for column in feature_columns:
        value = row[column]

        features[column] = (
            None if pd.isna(value) else float(value)
        )

    timestamp = pd.Timestamp(
        row["transaction_timestamp"]
    )

    return {
        "transaction_id": str(row["transaction_id"]),
        "transaction_timestamp": timestamp.isoformat(),
        "features": features,
    }


def send_score_request(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """POST a transaction-scoring request to the local API.

    Args:
        payload: JSON-compatible transaction and feature data.

    Returns:
        Parsed JSON response from the scoring endpoint.

    Raises:
        RuntimeError:
            If the API returns a non-success status or cannot be reached.
    """
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:
            response_body = response.read().decode("utf-8")

            if response.status != 200:
                raise RuntimeError(
                    f"API returned HTTP {response.status}: "
                    f"{response_body}"
                )

            return json.loads(response_body)

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")

        raise RuntimeError(
            f"API returned HTTP {exc.code}: {error_body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to the scoring API at {API_URL}. "
            "Make sure Uvicorn is running."
        ) from exc


def main() -> None:
    """Submit one real transaction and verify the persisted response."""
    payload = load_validation_request()

    print("Shepherd API integration smoke test")
    print("=" * 45)
    print(f"Transaction ID: {payload['transaction_id']}")
    print(f"Endpoint: {API_URL}")
    print("Submitting scoring request...")

    response = send_score_request(payload)

    required_response_fields = {
        "prediction_id",
        "transaction_id",
        "model_version",
        "fraud_probability",
        "threshold",
        "decision",
        "alert_triggered",
        "scored_at_utc",
    }

    missing_response_fields = (
        required_response_fields - set(response)
    )

    if missing_response_fields:
        raise RuntimeError(
            "Response is missing required fields: "
            f"{sorted(missing_response_fields)}"
        )

    if response["transaction_id"] != payload["transaction_id"]:
        raise RuntimeError(
            "Response transaction ID does not match the request."
        )

    print("\nPersisted scoring response")
    print("-" * 45)
    print(json.dumps(response, indent=2))
    print("\nAPI integration smoke test passed.")


if __name__ == "__main__":
    main()