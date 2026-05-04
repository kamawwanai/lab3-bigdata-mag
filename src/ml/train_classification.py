from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_CLASSIFICATION,
    load_feature_table,
    resolve_delta_version,
    resolve_max_ml_rows,
    train_test_split_by_time,
)

logger = logging.getLogger("ml.classification")


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _scores(y_true, y_prob, y_pred) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _log_feature_importance(model_name: str, pipe: Pipeline, x_test, y_test) -> None:
    sample = x_test.sample(n=min(len(x_test), 3000), random_state=42)
    sample_y = y_test.loc[sample.index]
    perm = permutation_importance(pipe, sample, sample_y, n_repeats=5, random_state=42, n_jobs=1)
    importances = {
        feature: float(importance)
        for feature, importance in sorted(
            zip(sample.columns, perm.importances_mean, strict=False),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    }
    mlflow.log_dict(importances, f"feature_importance_{model_name}.json")


def run_training(
    feature_table: Path,
    tracking_uri: str,
    experiment: str,
    gold_features_version: int,
    max_rows: int | None = None,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    row_limit = resolve_max_ml_rows(max_rows)
    df = load_feature_table(feature_table, max_rows=row_limit)
    train_df, test_df = train_test_split_by_time(df)
    x_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_CLASSIFICATION]
    x_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET_CLASSIFICATION]

    models = {
        "logistic_regression": LogisticRegression(max_iter=200, random_state=42),
        "gb_classifier": GradientBoostingClassifier(random_state=42),
    }
    for name, model in models.items():
        with mlflow.start_run(run_name=f"classification_{name}"):
            pipe = Pipeline([("preprocessor", _build_preprocessor()), ("model", model)])
            pipe.fit(x_train, y_train)

            prob = pipe.predict_proba(x_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            metrics = _scores(y_test, prob, pred)

            mlflow.log_param("model_name", name)
            mlflow.log_param("target", TARGET_CLASSIFICATION)
            mlflow.log_param("threshold_minutes", 15)
            mlflow.log_param("gold_feature_table_version", gold_features_version)
            mlflow.log_param("ml_max_rows", row_limit if row_limit is not None else "all")
            mlflow.log_param("train_rows", len(x_train))
            mlflow.log_param("test_rows", len(x_test))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path=f"model_{name}")
            mlflow.log_text(json.dumps(metrics, indent=2), "metrics.json")
            _log_feature_importance(name, pipe, x_test, y_test)
            logger.info("Classification run %s: %s", name, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classification models on Gold feature table.")
    parser.add_argument("--feature-table", type=Path, default=Path("lakehouse") / "gold" / "features")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="flight_delays_classification")
    parser.add_argument("--gold-features-version", type=int, default=-1)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional ML row cap. Defaults to ML_MAX_ROWS env.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    gold_features_version = (
        args.gold_features_version if args.gold_features_version >= 0 else resolve_delta_version(args.feature_table)
    )
    run_training(args.feature_table, args.tracking_uri, args.experiment, gold_features_version, max_rows=args.max_rows)


if __name__ == "__main__":
    main()

