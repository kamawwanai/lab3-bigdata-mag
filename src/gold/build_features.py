from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger("gold.features")


FEATURE_COLUMNS = [
    "flight_key",
    "flight_date",
    "year",
    "month",
    "hour",
    "day_of_week",
    "season",
    "carrier",
    "origin",
    "dest",
    "route",
    "dep_delay",
    "arr_delay",
    "is_delayed_15",
]


@dataclass(frozen=True)
class FeatureTableResult:
    rows: int
    delta_version: int


def build_feature_frame(
    silver_path: Path,
    *,
    filter_year: int | None = None,
    filter_month: int | None = None,
) -> pl.LazyFrame:
    lf = pl.scan_delta(str(silver_path))
    if filter_year is not None:
        lf = lf.filter(pl.col("year") == int(filter_year))
    if filter_month is not None:
        lf = lf.filter(pl.col("month") == int(filter_month))
    return (
        lf.select(FEATURE_COLUMNS)
        .filter(pl.col("arr_delay").is_not_null())
        .with_columns(
            [
                pl.col("carrier").fill_null("UNK"),
                pl.col("origin").fill_null("UNK"),
                pl.col("dest").fill_null("UNK"),
                pl.col("route").fill_null("UNK_UNK"),
                pl.col("dep_delay").fill_null(0.0),
            ]
        )
    )


def build_feature_table(silver_path: Path) -> pl.DataFrame:
    return build_feature_frame(silver_path).collect()


def write_feature_table(df: pl.DataFrame, output_path: Path) -> int:
    output_path.mkdir(parents=True, exist_ok=True)
    write_deltalake(
        str(output_path),
        df.to_arrow(),
        mode="overwrite",
        partition_by=["year", "month"],
    )
    dt = DeltaTable(str(output_path))
    return int(dt.version())


def run_feature_table(silver_path: Path, output_path: Path) -> FeatureTableResult:
    if output_path.exists():
        shutil.rmtree(output_path)

    partitions = (
        pl.scan_delta(str(silver_path))
        .select(["year", "month"])
        .filter(pl.col("year").is_not_null() & pl.col("month").is_not_null())
        .unique()
        .sort(["year", "month"])
        .collect()
    )

    output_path.mkdir(parents=True, exist_ok=True)
    first_write_done = False
    rows = 0
    for row in partitions.iter_rows(named=True):
        year = int(row["year"])
        month = int(row["month"])
        logger.info("Gold feature batch %s-%02d", year, month)
        batch_df = build_feature_frame(silver_path, filter_year=year, filter_month=month).collect()
        if batch_df.is_empty():
            continue
        write_deltalake(
            str(output_path),
            batch_df.to_arrow(),
            mode="append" if first_write_done else "overwrite",
            partition_by=["year", "month"],
        )
        first_write_done = True
        rows += batch_df.height

    if not first_write_done:
        raise ValueError("Gold feature table is empty; nothing was written.")

    version = int(DeltaTable(str(output_path)).version())
    return FeatureTableResult(rows=rows, delta_version=version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gold feature table from Silver.")
    parser.add_argument("--silver", type=Path, default=Path("lakehouse") / "silver" / "flights")
    parser.add_argument("--output", type=Path, default=Path("lakehouse") / "gold" / "features")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    result = run_feature_table(args.silver, args.output)
    logger.info("Gold feature table built: rows=%s version=%s", result.rows, result.delta_version)


if __name__ == "__main__":
    main()

