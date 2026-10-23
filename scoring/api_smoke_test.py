
"""Verify scoring persistence and idempotent retries through the HTTP API.

A transaction is loaded from the validation period of the feature dataset.
The script sends two identical requests using the same Idempotency-Key.

The second response should:
    - Have the same prediction_id as the first response.
    - Match the original response body.
    - Include the Idempotency-Replayed: true header.

The fraud target is not loaded or submitted.

Usage:
    Start the API:
        python -m uvicorn scoring.api:app --host 127.0.0.1 --port 8000

    Run this script in another terminal:
        python -m scoring.api_smoke_test
"""

import json
import urllib.error
import urllib.request
import uuid
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
        JSON-compatible transaction identifier, timestamp, and feature values.
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
    idempotency_key: str,
) -> tuple[dict[str, Any], str | None]:
    """POST a request using an idempotency key.

    Args:
        payload: JSON-compatible transaction-scoring request.
        idempotency_key: Key reused for retries of this exact request.

    Returns:
        Parsed JSON response and the Idempotency-Replayed response header.

    Raises:
        RuntimeError:
            If the API cannot be reached or returns a non-success status.
    """
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": idempotency_key,
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
                    f"API returned HTTP {response.status}: {response_body}"
                )

            return (
                json.loads(response_body),
                response.headers.get("Idempotency-Replayed"),
            )

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"API returned HTTP {exc.code}: {error_body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to {API_URL}. "
            "Ensure that the API server is running."
        ) from exc


def main() -> None:
    """Verify that an identical retry returns the existing prediction."""
    payload = load_validation_request()

    # Generate a new key for this smoke-test run, then reuse it for the retry.
    idempotency_key = str(uuid.uuid4())

    print("Shepherd API idempotency smoke test")
    print("=" * 50)
    print(f"Transaction ID: {payload['transaction_id']}")
    print(f"Idempotency-Key: {idempotency_key}")
    print("\nSending the initial request...")

    first_body, first_replay_header = send_score_request(
        payload,
        idempotency_key,
    )

    print(f"First replay header: {first_replay_header}")

    print("\nSending the same request again...")
    second_body, second_replay_header = send_score_request(
        payload,
        idempotency_key,
    )

    print(f"Second replay header: {second_replay_header}")

    if first_body != second_body:
        raise AssertionError(
            "The retry response body differs from the original response."
        )

    if first_body["prediction_id"] != second_body["prediction_id"]:
        raise AssertionError(
            "The retry created or returned a different prediction record."
        )

    if first_replay_header != "false":
        raise AssertionError(
            "The initial request should not be marked as a replay."
        )

    if second_replay_header != "true":
        raise AssertionError(
            "The repeated request should be marked as a replay."
        )

    print("\nPersisted scoring response")
    print("-" * 50)
    print(json.dumps(second_body, indent=2))
    print("\nIdempotency smoke test passed.")


if __name__ == "__main__":
    main()
