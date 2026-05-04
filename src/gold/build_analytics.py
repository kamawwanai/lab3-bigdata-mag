from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl
from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger("gold.analytics")


def _aggregate_delay_by(lf: pl.LazyFrame, column: str, metric_type: str) -> pl.LazyFrame:
    return (
        lf.group_by(column)
        .agg(
            [
                pl.len().alias("flights_cnt"),
                pl.mean("arr_delay").alias("avg_arr_delay"),
                pl.mean("dep_delay").alias("avg_dep_delay"),
                pl.mean("is_delayed_15").alias("share_delayed_15"),
            ]
        )
        .with_columns(pl.lit(metric_type).alias("metric_type"), pl.col(column).cast(pl.Utf8).alias("metric_key"))
        .select(["metric_type", "metric_key", "flights_cnt", "avg_arr_delay", "avg_dep_delay", "share_delayed_15"])
    )


def build_analytics(silver_path: Path) -> pl.DataFrame:
    lf = (
        pl.scan_delta(str(silver_path))
        .filter(pl.col("arr_delay").is_not_null())
        .select(["origin", "carrier", "hour", "season", "arr_delay", "dep_delay", "is_delayed_15"])
    )

    return pl.concat(
        [
            _aggregate_delay_by(lf, "origin", "airport").collect(),
            _aggregate_delay_by(lf, "carrier", "carrier").collect(),
            _aggregate_delay_by(lf, "hour", "hour").collect(),
            _aggregate_delay_by(lf, "season", "season").collect(),
        ],
        how="vertical_relaxed",
    )


def write_analytics(df: pl.DataFrame, output_path: Path) -> int:
    output_path.mkdir(parents=True, exist_ok=True)
    write_deltalake(str(output_path), df.to_arrow(), mode="overwrite")
    dt = DeltaTable(str(output_path))
    return int(dt.version())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gold analytics mart from Silver.")
    parser.add_argument("--silver", type=Path, default=Path("lakehouse") / "silver" / "flights")
    parser.add_argument("--output", type=Path, default=Path("lakehouse") / "gold" / "analytics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    df = build_analytics(args.silver)
    version = write_analytics(df, args.output)
    logger.info("Gold analytics built: rows=%s version=%s", df.height, version)


if __name__ == "__main__":
    main()

