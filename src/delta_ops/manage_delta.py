from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger("delta.ops")


def read_time_travel(path: Path, version: int) -> pl.DataFrame:
    dt = DeltaTable(str(path), version=version)
    return pl.from_arrow(dt.to_pyarrow_table())


def compact_table(path: Path) -> dict[str, Any]:
    dt = DeltaTable(str(path))
    if hasattr(dt, "optimize"):
        result = dt.optimize.compact()  # type: ignore[attr-defined]
        return {"method": "optimize.compact", "result": result}
    # Fallback: rewrite the current snapshot as one overwrite commit.
    df = pl.from_arrow(dt.to_pyarrow_table())
    write_deltalake(str(path), df.to_arrow(), mode="overwrite")
    return {"method": "overwrite_rewrite", "rows": df.height}


def vacuum_table(path: Path, dry_run: bool = True, retention_hours: int = 168) -> dict[str, Any]:
    dt = DeltaTable(str(path))
    if hasattr(dt, "vacuum"):
        deleted = dt.vacuum(retention_hours=retention_hours, dry_run=dry_run)
        return {"dry_run": dry_run, "retention_hours": retention_hours, "files": deleted}
    return {"dry_run": dry_run, "retention_hours": retention_hours, "files": []}


def z_order_table(path: Path, columns: list[str]) -> dict[str, Any]:
    dt = DeltaTable(str(path))
    cols = [c for c in columns if c]
    if not cols:
        return {"method": "skipped", "reason": "no_columns"}

    if hasattr(dt, "optimize") and hasattr(dt.optimize, "z_order"):
        result = dt.optimize.z_order(cols)  # type: ignore[attr-defined]
        return {"method": "optimize.z_order", "columns": cols, "result": result}

    df = pl.from_arrow(dt.to_pyarrow_table())
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return {"method": "skipped", "reason": "columns_not_found", "columns": cols}

    rewritten = df.sort(existing)
    write_deltalake(str(path), rewritten.to_arrow(), mode="overwrite")
    return {"method": "sort_overwrite_fallback", "columns": existing, "rows": rewritten.height}


def apply_schema_evolution(path: Path, new_column: str = "pipeline_run_id") -> int:
    dt = DeltaTable(str(path))
    df = pl.from_arrow(dt.to_pyarrow_table())
    if new_column in df.columns:
        return int(dt.version())
    evolved = df.with_columns(pl.lit("init").alias(new_column))
    write_deltalake(str(path), evolved.to_arrow(), mode="overwrite", schema_mode="overwrite")
    return int(DeltaTable(str(path)).version())


def main() -> None:
    parser = argparse.ArgumentParser(description="Delta operations: time travel, compaction, vacuum, schema evolution.")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--time-travel-version", type=int, default=0)
    parser.add_argument(
        "--zorder-cols",
        type=str,
        default="year,month,origin,dest,carrier",
        help="Comma-separated columns for Z-ORDER / clustering (best-effort).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    tt_df = read_time_travel(args.table, args.time_travel_version)
    compact_info = compact_table(args.table)
    zorder_cols = [c.strip() for c in str(args.zorder_cols).split(",") if c.strip()]
    zorder_info = z_order_table(args.table, zorder_cols)
    vacuum_info = vacuum_table(args.table, dry_run=True)
    new_version = apply_schema_evolution(args.table)

    payload = {
        "time_travel_rows": tt_df.height,
        "compaction": compact_info,
        "z_order": zorder_info,
        "vacuum": vacuum_info,
        "new_version_after_schema_evolution": new_version,
    }
    logger.info(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

