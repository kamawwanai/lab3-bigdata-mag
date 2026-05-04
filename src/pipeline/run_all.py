from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from deltalake import DeltaTable

from src.bronze.load_bronze import load_bronze_append_by_year, resolve_input_path
from src.gold.build_analytics import build_analytics, write_analytics
from src.gold.build_features import run_feature_table
from src.ml.train_classification import run_training as run_classification
from src.ml.train_regression import run_training as run_regression
from src.silver.transform_silver import run_silver

logger = logging.getLogger("pipeline")


def run_all(
    input_path: Path | None,
    bronze_path: Path,
    silver_path: Path,
    gold_analytics_path: Path,
    gold_features_path: Path,
    mlflow_uri: str,
    max_year: int | None = None,
    ml_max_rows: int | None = None,
) -> None:
    bronze = load_bronze_append_by_year(
        input_path=input_path,
        output_delta=bronze_path,
        reset_output=False,
        max_year=max_year,
    )
    logger.info("Bronze done: %s", bronze)

    silver = run_silver(bronze_path=bronze_path, silver_path=silver_path, max_year=max_year)
    logger.info("Silver done: %s", silver)

    analytics_df = build_analytics(silver_path)
    analytics_version = write_analytics(analytics_df, gold_analytics_path)
    logger.info("Gold analytics done. version=%s rows=%s", analytics_version, analytics_df.height)

    feature_result = run_feature_table(silver_path, gold_features_path)
    features_version = feature_result.delta_version
    logger.info("Gold features done. version=%s rows=%s", features_version, feature_result.rows)

    run_regression(
        feature_table=gold_features_path,
        tracking_uri=mlflow_uri,
        experiment="flight_delays_regression",
        gold_features_version=features_version,
        max_rows=ml_max_rows,
    )
    run_classification(
        feature_table=gold_features_path,
        tracking_uri=mlflow_uri,
        experiment="flight_delays_classification",
        gold_features_version=features_version,
        max_rows=ml_max_rows,
    )

    logger.info(
        "Pipeline completed. versions: bronze=%s silver=%s gold_analytics=%s gold_features=%s",
        DeltaTable(str(bronze_path)).version(),
        DeltaTable(str(silver_path)).version(),
        DeltaTable(str(gold_analytics_path)).version(),
        DeltaTable(str(gold_features_path)).version(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full lakehouse pipeline: bronze -> silver -> gold -> ml")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to source CSV/parquet file. Defaults to data/Flight_Delay.csv or data/Flight_Delay.parquet.",
    )
    parser.add_argument("--bronze", type=Path, default=Path("lakehouse") / "bronze" / "flights")
    parser.add_argument("--silver", type=Path, default=Path("lakehouse") / "silver" / "flights")
    parser.add_argument("--gold-analytics", type=Path, default=Path("lakehouse") / "gold" / "analytics")
    parser.add_argument("--gold-features", type=Path, default=Path("lakehouse") / "gold" / "features")
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--max-year", type=int, default=None, help="Optional upper bound for year (e.g. 2022).")
    parser.add_argument("--ml-max-rows", type=int, default=None, help="Optional ML row cap. Defaults to ML_MAX_ROWS env.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    ml_max_rows = args.ml_max_rows
    if ml_max_rows is None and os.getenv("ML_MAX_ROWS"):
        ml_max_rows = int(os.environ["ML_MAX_ROWS"])
    run_all(
        input_path=resolve_input_path(args.input),
        bronze_path=args.bronze,
        silver_path=args.silver,
        gold_analytics_path=args.gold_analytics,
        gold_features_path=args.gold_features,
        mlflow_uri=args.mlflow_uri,
        max_year=args.max_year,
        ml_max_rows=ml_max_rows,
    )


if __name__ == "__main__":
    main()

