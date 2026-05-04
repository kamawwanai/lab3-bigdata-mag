from __future__ import annotations

import argparse

import polars as pl


def build_query(table_path: str, year: int, month: int, min_arr_delay: float) -> pl.LazyFrame:
    return (
        pl.scan_delta(table_path)
        .filter(
            (pl.col("year") == year)
            & (pl.col("month") == month)
            & (pl.col("arr_delay") > min_arr_delay)
        )
        .select(["carrier", "arr_delay"])
        .group_by("carrier")
        .agg(pl.mean("arr_delay").alias("avg_arr_delay"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print Polars explain plan with visible PROJECT/SELECTION pushdowns."
    )
    parser.add_argument("--table", default="lakehouse/silver/flights", help="Path to Delta table")
    parser.add_argument("--year", type=int, default=2018, help="Year filter")
    parser.add_argument("--month", type=int, default=1, help="Month filter")
    parser.add_argument("--min-arr-delay", type=float, default=15.0, help="Minimum ARR_DELAY filter")
    args = parser.parse_args()

    lf = build_query(
        table_path=args.table,
        year=args.year,
        month=args.month,
        min_arr_delay=args.min_arr_delay,
    )

    print("=== EXPLAIN PLAN (optimized) ===")
    print(lf.explain(optimized=True))


if __name__ == "__main__":
    main()
