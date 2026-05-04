from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl
from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger("silver")


@dataclass(frozen=True)
class SilverBuildResult:
    input_rows: int
    output_rows: int
    written_mode: str
    delta_version: int


def _resolve_col(columns: Iterable[str], candidates: list[str], required: bool = True) -> str | None:
    lower_to_real = {c.lower(): c for c in columns}
    for cand in candidates:
        real = lower_to_real.get(cand.lower())
        if real is not None:
            return real
    if required:
        raise ValueError(f"None of columns {candidates} found. Available={list(columns)}")
    return None


def _season_expr(month_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(month_expr.is_in([12, 1, 2]))
        .then(pl.lit("winter"))
        .when(month_expr.is_in([3, 4, 5]))
        .then(pl.lit("spring"))
        .when(month_expr.is_in([6, 7, 8]))
        .then(pl.lit("summer"))
        .otherwise(pl.lit("autumn"))
    )


def _build_business_key(
    year_col: str,
    month_col: str,
    day_col: str | None,
    carrier_col: str | None,
    flight_num_col: str | None,
    origin_col: str | None,
    dest_col: str | None,
    dep_time_col: str | None,
) -> pl.Expr:
    parts: list[pl.Expr] = [
        pl.col(year_col).cast(pl.Utf8),
        pl.col(month_col).cast(pl.Utf8),
    ]
    optional_cols = [day_col, carrier_col, flight_num_col, origin_col, dest_col, dep_time_col]
    for col in optional_cols:
        if col is not None:
            parts.append(pl.col(col).cast(pl.Utf8))
    if len(parts) < 3:
        # Minimal fallback if many columns are missing.
        parts.append(pl.arange(0, pl.len()).cast(pl.Utf8))
    return pl.concat_str(parts, separator="|")


def _build_row_hash_key(columns: list[str]) -> pl.Expr:
    # Stable key even if the dataset lacks flight-identifying columns.
    wanted = [
        "year",
        "month",
        "day_of_month",
        "carrier",
        "flight_num",
        "origin",
        "dest",
        "dep_time_raw",
        "dep_delay",
        "arr_delay",
    ]
    present = [c for c in wanted if c in columns]
    if not present:
        return pl.arange(0, pl.len()).cast(pl.Utf8)
    return pl.struct([pl.col(c) for c in present]).hash().cast(pl.Utf8)


def build_silver_frame(
    bronze_path: Path,
    *,
    max_year: int | None = None,
    filter_year: int | None = None,
    filter_month: int | None = None,
) -> pl.LazyFrame:
    lf = pl.scan_delta(str(bronze_path))
    cols = lf.collect_schema().names()

    year_col = _resolve_col(cols, ["year"])
    month_col = _resolve_col(cols, ["month"])
    day_col = _resolve_col(cols, ["dayofmonth", "day_of_month", "day"], required=False)
    dep_delay_col = _resolve_col(cols, ["dep_delay", "depdelay"], required=False)
    arr_delay_col = _resolve_col(cols, ["arr_delay", "arrdelay"])
    cancelled_col = _resolve_col(cols, ["cancelled", "canceled"], required=False)
    carrier_col = _resolve_col(
        cols,
        ["carrier", "op_unique_carrier", "unique_carrier", "marketing_airline_network"],
        required=False,
    )
    flight_num_col = _resolve_col(cols, ["flight_num", "op_carrier_fl_num", "flight_number"], required=False)
    origin_col = _resolve_col(cols, ["origin", "origin_airport", "origincityname"], required=False)
    dest_col = _resolve_col(cols, ["dest", "destination", "dest_airport", "destcityname"], required=False)
    dep_time_col = _resolve_col(cols, ["dep_time", "deptime", "crs_dep_time", "crsdeptime"], required=False)

    lf = lf.with_columns(
        [
            pl.col(year_col).cast(pl.Int32, strict=False).alias("year"),
            pl.col(month_col).cast(pl.Int32, strict=False).alias("month"),
            pl.col(arr_delay_col).cast(pl.Float64, strict=False).alias("arr_delay"),
            (
                pl.col(dep_delay_col).cast(pl.Float64, strict=False).alias("dep_delay")
                if dep_delay_col
                else pl.lit(None).cast(pl.Float64).alias("dep_delay")
            ),
            (
                pl.col(cancelled_col).cast(pl.Int8, strict=False).alias("cancelled")
                if cancelled_col
                else pl.lit(0).cast(pl.Int8).alias("cancelled")
            ),
            (pl.col(carrier_col).cast(pl.Utf8).str.to_uppercase().alias("carrier") if carrier_col else pl.lit("UNK").alias("carrier")),
            (pl.col(origin_col).cast(pl.Utf8).str.to_uppercase().alias("origin") if origin_col else pl.lit("UNK").alias("origin")),
            (pl.col(dest_col).cast(pl.Utf8).str.to_uppercase().alias("dest") if dest_col else pl.lit("UNK").alias("dest")),
            (pl.col(flight_num_col).cast(pl.Utf8).alias("flight_num") if flight_num_col else pl.lit("UNK").alias("flight_num")),
            (pl.col(dep_time_col).cast(pl.Int32, strict=False).alias("dep_time_raw") if dep_time_col else pl.lit(None).cast(pl.Int32).alias("dep_time_raw")),
            (pl.col(day_col).cast(pl.Int32, strict=False).alias("day_of_month") if day_col else pl.lit(1).cast(pl.Int32).alias("day_of_month")),
        ]
    )

    date_expr = (
        pl.concat_str(
            [
                pl.col("year").cast(pl.Utf8),
                pl.col("month").cast(pl.Utf8).str.zfill(2),
                pl.col("day_of_month").cast(pl.Utf8).str.zfill(2),
            ],
            separator="-",
        )
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    )

    business_key = _build_business_key(
        year_col="year",
        month_col="month",
        day_col="day_of_month",
        carrier_col="carrier",
        flight_num_col="flight_num",
        origin_col="origin",
        dest_col="dest",
        dep_time_col="dep_time_raw",
    ).alias("flight_key")
    has_business_key = all(
        col is not None for col in [day_col, carrier_col, flight_num_col, origin_col, dest_col]
    )

    if max_year is not None:
        lf = lf.filter(pl.col("year").cast(pl.Int32, strict=False) <= int(max_year))
    if filter_year is not None:
        lf = lf.filter(pl.col("year") == int(filter_year))
    if filter_month is not None:
        lf = lf.filter(pl.col("month") == int(filter_month))

    # If the source doesn't have enough identifying columns, fall back to a stable row-hash key.
    # The preferred business key intentionally excludes target values, so changed delays update
    # existing rows instead of creating duplicates during MERGE.
    stable_key = _build_row_hash_key(
        [
            "year",
            "month",
            "day_of_month",
            "carrier",
            "flight_num",
            "origin",
            "dest",
            "dep_time_raw",
            "dep_delay",
            "arr_delay",
        ]
    ).alias("flight_key")
    key_expr = business_key if has_business_key else stable_key

    lf = (
        lf.filter(pl.col("cancelled") == 0)
        .filter(pl.col("year").is_not_null() & pl.col("month").is_not_null() & pl.col("arr_delay").is_not_null())
        .filter(pl.col("month").is_between(1, 12) & pl.col("day_of_month").is_between(1, 31))
        .filter(pl.col("arr_delay").is_between(-120.0, 360.0))
        .with_columns(
            [
                key_expr,
                pl.col("dep_time_raw").fill_null(0).cast(pl.Int32).alias("dep_time_raw"),
            ]
        )
        .with_columns(
            [
                (pl.col("dep_time_raw") // 100).clip(0, 23).alias("hour"),
                date_expr.alias("flight_date"),
            ]
        )
        .with_columns(
            [
                pl.col("flight_date").dt.weekday().alias("day_of_week"),
                _season_expr(pl.col("month")).alias("season"),
                pl.concat_str([pl.col("origin"), pl.lit("_"), pl.col("dest")]).alias("route"),
                (pl.col("arr_delay") > 15.0).cast(pl.Int8).alias("is_delayed_15"),
            ]
        )
        .filter(pl.col("flight_date").is_not_null())
        .select(
            [
                "flight_key",
                "flight_date",
                "year",
                "month",
                "day_of_month",
                "hour",
                "day_of_week",
                "season",
                "carrier",
                "flight_num",
                "origin",
                "dest",
                "route",
                "dep_delay",
                "arr_delay",
                "is_delayed_15",
            ]
        )
        .unique(subset=["flight_key"], keep="last")
    )
    return lf


def _delta_version(path: Path) -> int:
    dt = DeltaTable(str(path))
    return int(dt.version())


def _existing_year_months(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()

    try:
        existing = pl.scan_delta(str(path))
        cols = existing.collect_schema().names()
        year_col = _resolve_col(cols, ["year"])
        month_col = _resolve_col(cols, ["month"])
        partitions = (
            existing.select(
                [
                    pl.col(year_col).cast(pl.Int32, strict=False).alias("year"),
                    pl.col(month_col).cast(pl.Int32, strict=False).alias("month"),
                ]
            )
            .filter(pl.col("year").is_not_null() & pl.col("month").is_not_null())
            .unique()
            .collect(streaming=True)
        )
        return {(int(row["year"]), int(row["month"])) for row in partitions.iter_rows(named=True)}
    except Exception:
        logger.warning("Could not inspect existing Silver table; all batches will be processed.", exc_info=True)
        return set()


def write_silver_batch(silver_df: pl.DataFrame, silver_path: Path, *, use_merge: bool) -> str:
    table = silver_df.to_arrow()
    if not silver_path.exists():
        silver_path.mkdir(parents=True, exist_ok=True)
        write_deltalake(
            str(silver_path),
            table,
            mode="overwrite",
            partition_by=["year", "month"],
        )
        return "overwrite"

    try:
        dt = DeltaTable(str(silver_path))
    except Exception:
        logger.warning("Existing Silver path is not a readable Delta table. Recreating: %s", silver_path)
        shutil.rmtree(silver_path)
        silver_path.mkdir(parents=True, exist_ok=True)
        write_deltalake(
            str(silver_path),
            table,
            mode="overwrite",
            partition_by=["year", "month"],
        )
        return "overwrite"

    if use_merge:
        (
            dt.merge(
                source=table,
                predicate="target.flight_key = source.flight_key",
                source_alias="source",
                target_alias="target",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
        return "merge"

    write_deltalake(
        str(silver_path),
        table,
        mode="append",
        partition_by=["year", "month"],
    )
    return "append"


def run_silver(bronze_path: Path, silver_path: Path, *, max_year: int | None = None) -> SilverBuildResult:
    src_lf = pl.scan_delta(str(bronze_path))
    cols = src_lf.collect_schema().names()
    year_col = _resolve_col(cols, ["year"])
    month_col = _resolve_col(cols, ["month"])
    if max_year is not None:
        src_lf = src_lf.filter(pl.col(year_col).cast(pl.Int32, strict=False) <= int(max_year))
    input_rows = int(src_lf.select(pl.len()).collect().item())

    partitions = (
        src_lf.select(
            [
                pl.col(year_col).cast(pl.Int32, strict=False).alias("year"),
                pl.col(month_col).cast(pl.Int32, strict=False).alias("month"),
            ]
        )
        .filter(pl.col("year").is_not_null() & pl.col("month").is_not_null())
        .unique()
        .sort(["year", "month"])
        .collect()
    )

    output_rows = 0
    modes: set[str] = set()
    existing_batches = _existing_year_months(silver_path)
    for row in partitions.iter_rows(named=True):
        year = int(row["year"])
        month = int(row["month"])
        batch_exists = (year, month) in existing_batches
        logger.info("Silver batch %s-%02d (%s)", year, month, "merge" if batch_exists else "append")
        batch_df = build_silver_frame(
            bronze_path,
            max_year=max_year,
            filter_year=year,
            filter_month=month,
        ).collect()
        if batch_df.is_empty():
            continue
        modes.add(write_silver_batch(batch_df, silver_path, use_merge=batch_exists))
        if not batch_exists:
            existing_batches.add((year, month))
        output_rows += batch_df.height

    if not modes:
        if existing_batches:
            return SilverBuildResult(
                input_rows=input_rows,
                output_rows=0,
                written_mode="skip",
                delta_version=_delta_version(silver_path),
            )
        raise ValueError("Silver output is empty; nothing was written.")

    mode = "merge" if "merge" in modes else ("append" if "append" in modes else "overwrite")
    version = _delta_version(silver_path)
    return SilverBuildResult(
        input_rows=input_rows,
        output_rows=output_rows,
        written_mode=mode,
        delta_version=version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Silver Delta table from Bronze.")
    parser.add_argument("--bronze", type=Path, default=Path("lakehouse") / "bronze" / "flights")
    parser.add_argument("--silver", type=Path, default=Path("lakehouse") / "silver" / "flights")
    parser.add_argument("--max-year", type=int, default=None, help="Optional upper bound for year (e.g. 2022).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    result = run_silver(args.bronze, args.silver, max_year=args.max_year)
    logger.info(
        "Silver built: input_rows=%s output_rows=%s mode=%s version=%s",
        result.input_rows,
        result.output_rows,
        result.written_mode,
        result.delta_version,
    )


if __name__ == "__main__":
    main()

