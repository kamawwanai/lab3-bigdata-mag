from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl
from deltalake import DeltaTable, write_deltalake


logger = logging.getLogger("bronze")

DEFAULT_INPUT_CANDIDATES = (
    Path("data") / "Flight_Delay.csv",
    Path("data") / "Flight_Delay.parquet",
)


@dataclass(frozen=True)
class BronzeLoadResult:
    years_written: list[int]
    source_rows: int
    written_rows: int
    skipped_rows: int
    delta_versions_after: int


def _resolve_existing_column(columns: Iterable[str], wanted: str) -> str | None:
    wanted_l = wanted.lower()
    for c in columns:
        if c.lower() == wanted_l:
            return c
    return None


def _get_year_column_name(columns: list[str]) -> str:
    col = _resolve_existing_column(columns, "year")
    if col is None:
        raise ValueError(
            "Не найдена колонка year (case-insensitive). "
            f"Доступные колонки: {columns}"
        )
    return col


def _get_month_column_name(columns: list[str]) -> str | None:
    return _resolve_existing_column(columns, "month")


def _ensure_int_year(expr: pl.Expr) -> pl.Expr:
    # Handles ints, strings, and floats with possible NAs.
    return expr.cast(pl.Int32, strict=False)


def _delta_version_count(table_path: Path) -> int:
    if not table_path.exists():
        return 0
    try:
        dt = DeltaTable(str(table_path))
        # version is 0-based; number of versions = last_version + 1
        return int(dt.version()) + 1
    except Exception:
        return 0


def _existing_year_months(table_path: Path) -> set[tuple[int, int]]:
    if _delta_version_count(table_path) == 0:
        return set()

    try:
        existing = pl.scan_delta(str(table_path))
        cols = existing.collect_schema().names()
        year_col = _get_year_column_name(cols)
        month_col = _get_month_column_name(cols)
        if month_col is None:
            logger.warning("Existing Bronze table has no month column; append idempotency is disabled.")
            return set()

        batches_df = (
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
        return {(int(row["year"]), int(row["month"])) for row in batches_df.iter_rows(named=True)}
    except Exception:
        logger.warning("Could not inspect existing Bronze table; append idempotency is disabled.", exc_info=True)
        return set()


def resolve_input_path(input_path: Path | None = None) -> Path:
    if input_path is not None:
        if input_path.exists():
            return input_path
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    expected = ", ".join(str(p) for p in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"Input dataset not found. Expected one of: {expected}")


def scan_source(input_path: Path) -> pl.LazyFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(str(input_path), infer_schema_length=10000, ignore_errors=True)
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(str(input_path))
    raise ValueError(f"Unsupported input format: {input_path}. Expected .csv or .parquet")


def load_bronze_append_by_year(
    *,
    input_parquet: Path | None = None,
    input_path: Path | None = None,
    output_delta: Path,
    reset_output: bool = False,
    max_year: int | None = None,
) -> BronzeLoadResult:
    source_path = resolve_input_path(input_path or input_parquet)

    if reset_output and output_delta.exists():
        logger.warning("Reset enabled. Removing existing output: %s", output_delta)
        shutil.rmtree(output_delta)

    scan = scan_source(source_path)
    cols = scan.collect_schema().names()
    year_col = _get_year_column_name(cols)
    month_col = _get_month_column_name(cols)
    if month_col is None:
        raise ValueError(
            "Не найдена колонка month (case-insensitive). "
            "По запросу загрузка Bronze выполняется строго помесячно."
        )

    scan = scan.with_columns(_ensure_int_year(pl.col(year_col)).alias(year_col))
    if max_year is not None:
        scan = scan.filter(pl.col(year_col) <= int(max_year))

    years_df = (
        scan.select(pl.col(year_col).drop_nulls().unique().sort()).collect(streaming=True)
    )
    years = years_df.get_column(year_col).to_list()
    years = [int(y) for y in years]
    if not years:
        raise ValueError(f"Список year пустой после чтения {source_path}.")

    source_rows = int(scan.select(pl.len()).collect(streaming=True).item())

    existing_batches = _existing_year_months(output_delta)
    output_delta.mkdir(parents=True, exist_ok=True)

    written_rows = 0
    skipped_rows = 0
    years_written: list[int] = []

    first_write_done = _delta_version_count(output_delta) > 0

    for y in years:
        year_lf = scan.filter(pl.col(year_col) == y)
        year_rows = int(year_lf.select(pl.len()).collect(streaming=True).item())
        logger.info("Year %s: %s rows", y, year_rows)
        if year_rows == 0:
            continue

        months_df = (
            year_lf.select(pl.col(month_col).drop_nulls().cast(pl.Int32, strict=False).unique().sort())
            .collect(streaming=True)
        )
        months = [int(m) for m in months_df.get_column(month_col).to_list()]
        if not months:
            logger.warning("Year %s: список month пустой, пропускаю.", y)
            continue

        year_written = False
        for m in months:
            batch_lf = year_lf.filter(pl.col(month_col).cast(pl.Int32, strict=False) == m)
            if (y, m) in existing_batches:
                batch_rows = int(batch_lf.select(pl.len()).collect(streaming=True).item())
                logger.info("  %s-%02d already exists in Bronze, skipping append.", y, m)
                skipped_rows += batch_rows
                continue

            df = batch_lf.collect(streaming=True)
            batch_rows = df.height
            logger.info("  %s-%02d: %s rows", y, m, batch_rows)
            if batch_rows == 0:
                continue
            table = df.to_arrow()

            mode = "append" if first_write_done else "overwrite"
            write_deltalake(str(output_delta), table, mode=mode)
            first_write_done = True
            existing_batches.add((y, m))
            written_rows += batch_rows
            year_written = True

        if year_written:
            years_written.append(y)

    versions_after = _delta_version_count(output_delta)
    return BronzeLoadResult(
        years_written=years_written,
        source_rows=source_rows,
        written_rows=written_rows,
        skipped_rows=skipped_rows,
        delta_versions_after=versions_after,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bronze: CSV/parquet -> Delta (append batches by year and month)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to source CSV/parquet file. Defaults to data/Flight_Delay.csv or data/Flight_Delay.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lakehouse") / "bronze" / "flights",
        help="Path to Delta table directory",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete output directory before loading (DANGEROUS).",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Optional upper bound for year (e.g. 2022).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    result = load_bronze_append_by_year(
        input_path=args.input,
        output_delta=args.output,
        reset_output=args.reset,
        max_year=args.max_year,
    )

    logger.info(
        "Done. years=%s source_rows=%s written_rows=%s skipped_rows=%s delta_versions=%s",
        result.years_written,
        result.source_rows,
        result.written_rows,
        result.skipped_rows,
        result.delta_versions_after,
    )

    if result.source_rows != result.written_rows + result.skipped_rows:
        raise SystemExit(
            f"Row count mismatch: source_rows={result.source_rows} "
            f"written_rows={result.written_rows} skipped_rows={result.skipped_rows}"
        )

    # Expect at least one version, and ideally >= number of year batches written.
    if result.delta_versions_after < 1:
        raise SystemExit("Delta table has no versions after write (unexpected).")
    if result.delta_versions_after < len(result.years_written):
        logger.warning(
            "Delta versions (%s) < years written (%s). "
            "This can happen if multiple years were committed together, "
            "but should be investigated.",
            result.delta_versions_after,
            len(result.years_written),
        )


if __name__ == "__main__":
    main()

