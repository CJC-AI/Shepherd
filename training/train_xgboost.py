"""
Train the Shepherd baseline XGBoost fraud model.

This module loads the validated chronological dataset split, computes the
training class-imbalance weight, trains XGBoost using the validation period
for early stopping, evaluates the model on the untouched test period, and
persists the trained model together with the model feature list and metrics.

The test set is never used for model fitting or early stopping.
"""

import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from training.dataset_loader import (
    MODEL_FEATURES,
    load_feature_dataset,
    split_dataset,
)
from training.model_config import (
    DEFAULT_XGBOOST_CONFIG,
)


MODEL_DIR = Path(
    "artifacts/models"
)

MODEL_PATH = MODEL_DIR / "xgboost_baseline.json"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"


def calculate_scale_pos_weight(
    y_train: np.ndarray,
) -> float:
    """
    Calculate XGBoost's positive-class weight.

    Parameters
    ----------
    y_train:
        Binary training labels where fraud is represented by 1 and non-fraud
        by 0.

    Returns
    -------
    float
        Ratio of negative examples to positive examples.

    Raises
    ------
    ValueError
        If the training data contains no positive examples.
    """

    positive_count = int(
        np.sum(y_train == 1)
    )

    negative_count = int(
        np.sum(y_train == 0)
    )

    if positive_count == 0:
        raise ValueError(
            "Training data contains no positive fraud examples."
        )

    return negative_count / positive_count


def train_baseline():
    """
    Train and evaluate the baseline XGBoost model.

    Returns
    -------
    tuple
        Trained XGBoost classifier and evaluation metrics.
    """

    dataframe = load_feature_dataset()
    dataset = split_dataset(dataframe)

    scale_pos_weight = calculate_scale_pos_weight(
        dataset.y_train
    )

    config = DEFAULT_XGBOOST_CONFIG

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=config.random_state,
        early_stopping_rounds=(
            config.early_stopping_rounds
        ),
        n_jobs=-1,
    )

    print(
        f"Training rows: "
        f"{len(dataset.X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(dataset.X_validation):,}"
    )

    print(
        f"Test rows: "
        f"{len(dataset.X_test):,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    model.fit(
        dataset.X_train,
        dataset.y_train,
        eval_set=[
            (
                dataset.X_validation,
                dataset.y_validation,
            )
        ],
        verbose=50,
    )

    validation_probability = model.predict_proba(
        dataset.X_validation
    )[:, 1]

    test_probability = model.predict_proba(
        dataset.X_test
    )[:, 1]

    validation_pr_auc = (
        average_precision_score(
            dataset.y_validation,
            validation_probability,
        )
    )

    validation_roc_auc = (
        roc_auc_score(
            dataset.y_validation,
            validation_probability,
        )
    )

    test_pr_auc = (
        average_precision_score(
            dataset.y_test,
            test_probability,
        )
    )

    test_roc_auc = (
        roc_auc_score(
            dataset.y_test,
            test_probability,
        )
    )

    metrics = {
        "model": "xgboost_baseline",
        "feature_count": len(MODEL_FEATURES),
        "train_rows": len(dataset.X_train),
        "validation_rows": len(
            dataset.X_validation
        ),
        "test_rows": len(dataset.X_test),
        "train_fraud_rate": float(
            dataset.y_train.mean()
        ),
        "validation_fraud_rate": float(
            dataset.y_validation.mean()
        ),
        "test_fraud_rate": float(
            dataset.y_test.mean()
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "best_iteration": int(
            model.best_iteration
        ),
        "best_validation_pr_auc": float(
            model.best_score
        ),
        "validation_pr_auc": float(
            validation_pr_auc
        ),
        "validation_roc_auc": float(
            validation_roc_auc
        ),
        "test_pr_auc": float(
            test_pr_auc
        ),
        "test_roc_auc": float(
            test_roc_auc
        ),
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        MODEL_PATH
    )

    FEATURES_PATH.write_text(
        json.dumps(
            MODEL_FEATURES,
            indent=2,
        )
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print()
    print(
        "Baseline model training complete."
    )

    print(
        f"Best iteration: "
        f"{model.best_iteration}"
    )

    print(
        f"Validation PR-AUC: "
        f"{validation_pr_auc:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_roc_auc:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_pr_auc:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_roc_auc:.4f}"
    )

    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )

    return model, metrics


def main() -> None:
    """
    Train the baseline XGBoost model.

    Returns
    -------
    None
    """

    train_baseline()


if __name__ == "__main__":
    main()