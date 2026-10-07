"""
metrics.py — Deterministic metrics computation over partitioned Parquet
data, using DuckDB.

WHY DuckDB instead of plain pandas here:
DuckDB can query the Hive-partitioned directory structure directly via a
glob pattern (`year=*/month=*/day=*/orders.parquet`) and treats year/month/
day as queryable columns automatically, without manually walking folders
in Python. It also means the exact same SQL will run unmodified against
S3 later (DuckDB's httpfs extension reads s3:// paths with this same glob
syntax) — this is the payoff of the partitioning decision made above.

WHY these specific metrics, and not more:
This is intentionally a small, defensible metric set — daily revenue,
order count, AOV, units, unique customers, returns, top products — rather
than a large dashboard's worth of KPIs. In the original scoping, the
analytics are NOT meant to be the differentiator; overbuilding this layer
risks the 3-4 day budget on the least interview-relevant part of the
project. WoW/MoM deltas are computed because they're what the Day 2 LLM
narrative actually needs ("revenue is up/down vs comparable prior period")
— metrics are built to serve the report, not for their own sake.

ROUNDING NOTE: financial figures are rounded to 2 decimals only at the
final output boundary, never mid-calculation, to avoid compounding
rounding error across aggregations.
"""

import calendar
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import paths  # noqa: E402


def _sql_path(path: Path) -> str:
    """Forward-slash, quote-escaped path literal (safe on Windows and for paths containing quotes)."""
    return Path(path).as_posix().replace("'", "''")


def get_connection(partitioned_dir: Path) -> duckdb.DuckDBPyConnection:
    """
    In-memory DuckDB with a ``product_sales`` and a ``cancellations`` view over the
    Hive-partitioned parquet files. ``union_by_name`` keeps the views valid when
    partitions were written at different times, and a dataset with no partitions
    yet is an empty view rather than an error.
    """
    con = duckdb.connect(database=":memory:")
    for dataset in (paths.PRODUCT_SALES, paths.CANCELLATIONS):
        base = Path(partitioned_dir) / dataset
        if next(base.glob("year=*/month=*/day=*/orders.parquet"), None) is None:
            con.execute(
                f"CREATE VIEW {dataset} AS SELECT NULL::VARCHAR AS InvoiceNo, NULL::VARCHAR AS Description, "
                "NULL::BIGINT AS Quantity, NULL::DOUBLE AS UnitPrice, NULL::DOUBLE AS CustomerID, "
                "NULL::BIGINT AS year, NULL::BIGINT AS month, NULL::BIGINT AS day WHERE false"
            )
            continue
        glob = _sql_path(base / "year=*" / "month=*" / "day=*" / "orders.parquet")
        con.execute(
            f"CREATE VIEW {dataset} AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)"
        )
    return con


def compute_daily_metrics(con: duckdb.DuckDBPyConnection, target_date: date) -> dict:
    """
    Computes the full metric set for a single day, plus WoW comparison
    (same weekday, 7 days prior) and MoM comparison (same date, prior
    calendar month, falling back gracefully if that date has no data —
    real retail data has gaps, e.g. this dataset has no Dec 25 order).
    """
    d = target_date
    wow_d = d - timedelta(days=7)

    # BUGFIX (found during Day 2 anomaly-detection validation, which is the
    # first code path to call this for every day in the dataset rather than
    # a single hand-picked date): naively substituting d.month - 1 breaks
    # whenever the target day doesn't exist in the prior month (e.g.
    # "2011-03-31" -> "2011-02-31", which is not a valid date). Fixed by
    # clamping to the prior month's actual last day via calendar.monthrange,
    # which is the standard approach for calendar-safe month arithmetic.
    prior_month = d.month - 1 or 12
    prior_month_year = d.year if d.month > 1 else d.year - 1
    last_day_prior_month = calendar.monthrange(prior_month_year, prior_month)[1]
    mom_d = date(prior_month_year, prior_month, min(d.day, last_day_prior_month))

    def core_metrics(query_date: date) -> dict | None:
        row = con.execute("""
            SELECT
                COUNT(DISTINCT InvoiceNo)                          AS order_count,
                ROUND(SUM(Quantity * UnitPrice), 2)                AS revenue,
                SUM(Quantity)                                      AS units_sold,
                COUNT(DISTINCT CustomerID)                         AS unique_customers
            FROM product_sales
            WHERE make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) = ?
        """, [query_date]).fetchone()
        # Guard: no rows for this date (dataset has real gaps, e.g. weekends/holidays)
        if row is None or row[0] == 0:
            return None
        order_count, revenue, units_sold, unique_customers = row
        aov = round(revenue / order_count, 2) if order_count else 0.0
        return {
            "order_count": order_count,
            "revenue": float(revenue) if revenue else 0.0,
            "units_sold": int(units_sold) if units_sold else 0,
            "unique_customers": unique_customers,
            "aov": aov,
        }

    def return_metrics(query_date: date) -> dict:
        row = con.execute("""
            SELECT COUNT(DISTINCT InvoiceNo), ROUND(SUM(Quantity * UnitPrice), 2)
            FROM cancellations
            WHERE make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) = ?
        """, [query_date]).fetchone()
        count, value = row
        return {
            "return_count": count or 0,
            # value is negative (quantity is negative on cancellations); report as positive magnitude
            "return_value": abs(float(value)) if value else 0.0,
        }

    def top_products(query_date: date, limit: int = 5) -> list:
        rows = con.execute("""
            SELECT Description, ROUND(SUM(Quantity * UnitPrice), 2) AS rev
            FROM product_sales
            WHERE make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) = ?
            GROUP BY Description
            ORDER BY rev DESC
            LIMIT ?
        """, [query_date, limit]).fetchall()
        return [{"product": r[0], "revenue": float(r[1])} for r in rows]

    today = core_metrics(d)
    if today is None:
        return {"date": str(d), "has_data": False, "note": "No transactions recorded for this date."}

    wow = core_metrics(wow_d)
    mom = core_metrics(mom_d)
    returns = return_metrics(d)

    def pct_change(curr, prior):
        if prior in (None, 0):
            return None
        return round((curr - prior) / prior * 100, 1)

    result = {
        "date": str(d),
        "has_data": True,
        **today,
        **returns,
        "top_products": top_products(d),
        "comparisons": {
            "wow": {
                "reference_date": str(wow_d),
                "reference_available": wow is not None,
                "revenue_pct_change": pct_change(today["revenue"], wow["revenue"]) if wow else None,
                "order_count_pct_change": pct_change(today["order_count"], wow["order_count"]) if wow else None,
            },
            "mom": {
                "reference_date": str(mom_d),
                "reference_available": mom is not None,
                "revenue_pct_change": pct_change(today["revenue"], mom["revenue"]) if mom else None,
                "order_count_pct_change": pct_change(today["order_count"], mom["order_count"]) if mom else None,
            },
        },
    }
    return result


if __name__ == "__main__":
    root = paths.PROJECT_ROOT
    con = get_connection(paths.partitioned_dir(root))

    # Demo: compute metrics for the last real date in the dataset, so the
    # WoW comparison has real prior data to compare against.
    target = date(2011, 12, 9)
    metrics = compute_daily_metrics(con, target)
    print(json.dumps(metrics, indent=2))

    out_dir = paths.reports_dir(root)
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"metrics_{target}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
