"""
history_cache.py - incremental Parquet cache for the daily-history DataFrame that
anomaly_detection.py uses as its statistical baseline.

Recomputing every day's metrics on each call gets slower forever as history grows
and is wasteful for a Lambda that runs once a day. This module caches one row per
day in a small Parquet file and only computes days that are not cached yet, so a
steady-state daily run touches exactly one new partition regardless of history
size. Missing days are computed together in a single DuckDB query, not one query
per metric per day.

The cache is a plain local path on purpose: the Lambda handler syncs it to and
from S3, keeping S3 out of this module so the caching logic stays unit-testable.
"""

import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import paths  # noqa: E402
from pipeline.metrics import compute_daily_metrics  # noqa: E402

HISTORY_COLUMNS = ["date", "revenue", "order_count", "return_count", "return_value"]
_DTYPES = {
    "date": "object",
    "revenue": "float64",
    "order_count": "int64",
    "return_count": "int64",
    "return_value": "float64",
}
_QUERY_CHUNK = 400  # dates per query, keeps the IN-list small


def _empty_history() -> pd.DataFrame:
    # Explicit dtypes: an untyped empty frame is all-object and would poison the
    # dtypes of later concatenation, making cold and warm runs differ.
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in _DTYPES.items()})


def _compute_row(con: duckdb.DuckDBPyConnection, d: date) -> dict | None:
    """Metrics row for a single day (None when the day has no sales)."""
    m = compute_daily_metrics(con, d)
    if not m.get("has_data"):
        return None
    return {
        "date": d,
        "revenue": m["revenue"],
        "order_count": m["order_count"],
        "return_count": m["return_count"],
        "return_value": m["return_value"],
    }


def _all_available_dates(con: duckdb.DuckDBPyConnection) -> list[date]:
    rows = con.execute(
        "SELECT DISTINCT make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) AS d "
        "FROM product_sales ORDER BY d"
    ).fetchall()
    return [row[0] for row in rows]


def _compute_rows(con: duckdb.DuckDBPyConnection, dates: list[date]) -> pd.DataFrame:
    """Daily rows for ``dates`` using one grouped query per chunk (same maths as compute_daily_metrics)."""
    frames = []
    for start in range(0, len(dates), _QUERY_CHUNK):
        chunk = dates[start:start + _QUERY_CHUNK]
        keys = ", ".join(f"({d.year}, {d.month}, {d.day})" for d in chunk)
        frames.append(
            con.execute(
                f"""
                WITH s AS (
                    SELECT make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) AS d,
                           COUNT(DISTINCT InvoiceNo) AS order_count,
                           ROUND(SUM(Quantity * UnitPrice), 2) AS revenue
                    FROM product_sales
                    WHERE (CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) IN ({keys})
                    GROUP BY d
                ),
                c AS (
                    SELECT make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) AS d,
                           COUNT(DISTINCT InvoiceNo) AS return_count,
                           ABS(ROUND(SUM(Quantity * UnitPrice), 2)) AS return_value
                    FROM cancellations
                    WHERE (CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) IN ({keys})
                    GROUP BY d
                )
                SELECT s.d AS date, COALESCE(s.revenue, 0.0) AS revenue, s.order_count,
                       COALESCE(c.return_count, 0) AS return_count, COALESCE(c.return_value, 0.0) AS return_value
                FROM s LEFT JOIN c USING (d)
                WHERE s.order_count > 0
                ORDER BY s.d
                """
            ).df()
        )
    if not frames:
        return _empty_history()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].map(lambda v: pd.Timestamp(v).date())
    return out.astype({k: v for k, v in _DTYPES.items() if k != "date"})[HISTORY_COLUMNS]


def compute_history(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Daily metrics for every date in product_sales, without touching any cache."""
    dates = _all_available_dates(con)
    return _compute_rows(con, dates) if dates else _empty_history()


def get_history(con: duckdb.DuckDBPyConnection, cache_path: Path, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Full daily-history DataFrame, backed by the Parquet cache at ``cache_path``.
    Only dates missing from the cache are computed; the cache is rewritten when it
    grows. ``force_rebuild=True`` ignores the cache (e.g. after a cleaning change).
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        cached = pd.read_parquet(cache_path)
        cached["date"] = pd.to_datetime(cached["date"]).dt.date
    else:
        cached = _empty_history()

    cached_dates = set(cached["date"])
    missing = [d for d in _all_available_dates(con) if d not in cached_dates]
    if not missing and not cached.empty:
        return cached.sort_values("date").reset_index(drop=True)

    new_rows = _compute_rows(con, missing) if missing else _empty_history()
    updated = new_rows if cached.empty else (cached if new_rows.empty else pd.concat([cached, new_rows], ignore_index=True))
    updated = updated.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(cache_path, index=False)
    return updated


if __name__ == "__main__":
    import time

    from pipeline.metrics import get_connection

    root = paths.PROJECT_ROOT
    con = get_connection(paths.partitioned_dir(root))
    cache = paths.history_cache_path(root)
    cache.unlink(missing_ok=True)

    t0 = time.time()
    cold = get_history(con, cache)
    print(f"Cold run (no cache): {len(cold)} rows in {time.time() - t0:.2f}s")
    t0 = time.time()
    warm = get_history(con, cache)
    print(f"Warm run (fully cached): {len(warm)} rows in {time.time() - t0:.2f}s")
    assert cold.equals(warm), "Cold and warm run produced different data"
    print("Cold/warm consistency check: PASSED")
