"""
Inspect the trained Shepherd XGBoost baseline model.

This module loads the persisted baseline model and reports feature importance
using XGBoost's gain-based importance measure.

The purpose is diagnostic: understand which engineered features the baseline
model is relying on before performing threshold tuning or hyperparameter
optimization.
"""

import json
from pathlib import Path

import xgboost as xgb

from training.dataset_loader import MODEL_FEATURES


MODEL_PATH = Path(
    "artifacts/models/xgboost_baseline.json"
)


def inspect_model() -> None:
    """
    Load the baseline model and print feature importance.

    Returns
    -------
    None
    """

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    booster = model.get_booster()

    gain_importance = booster.get_score(
        importance_type="gain"
    )

    rows = []

    for feature_index, feature_name in enumerate(
        MODEL_FEATURES
    ):
        booster_name = f"f{feature_index}"

        rows.append(
            (
                feature_name,
                gain_importance.get(
                    booster_name,
                    0.0,
                ),
            )
        )

    rows.sort(
        key=lambda row: row[1],
        reverse=True,
    )

    total_gain = sum(
        importance
        for _, importance in rows
    )

    print("Feature importance by gain:")
    print()

    for rank, (feature, gain) in enumerate(
        rows,
        start=1,
    ):
        share = (
            gain / total_gain
            if total_gain > 0
            else 0.0
        )

        print(
            f"{rank:2d}. "
            f"{feature:<35} "
            f"gain={gain:,.4f} "
            f"share={share:.2%}"
        )


def main() -> None:
    """
    Inspect the baseline model.

    Returns
    -------
    None
    """

    inspect_model()


if __name__ == "__main__":
    main()