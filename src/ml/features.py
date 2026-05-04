from __future__ import annotations

import os
from pathlib import Path

from deltalake import DeltaTable
import pandas as pd
import polars as pl


NUMERIC_FEATURES = ["year", "month", "hour", "day_of_week", "dep_delay"]
CATEGORICAL_FEATURES = ["season", "carrier", "origin", "dest", "route"]
TARGET_REGRESSION = "arr_delay"
TARGET_CLASSIFICATION = "is_delayed_15"
DEFAULT_MAX_ML_ROWS = 300_000


def resolve_max_ml_rows(max_rows: int | None = None) -> int | None:
    if max_rows is not None:
        return max_rows if max_rows > 0 else None
    raw = os.getenv("ML_MAX_ROWS", str(DEFAULT_MAX_ML_ROWS)).strip()
    if not raw:
        return None
    value = int(raw)
    return value if value > 0 else None


def load_feature_table(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    row_limit = resolve_max_ml_rows(max_rows)
    lf = pl.scan_delta(str(path))
    if row_limit is not None:
        lf = lf.limit(row_limit)
    df = lf.collect()
    return df.to_pandas()


def resolve_delta_version(path: Path) -> int:
    return int(DeltaTable(str(path)).version())


def train_test_split_by_time(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("flight_date")
    split_idx = int(len(df) * (1.0 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

