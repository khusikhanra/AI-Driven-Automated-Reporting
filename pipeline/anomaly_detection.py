"""
anomaly_detection.py — Statistical anomaly detection and rule-based
business-event flags, built as a separate module from Day 1's metrics.py
(Day 1 stays untouched; this module reads from the same DuckDB connection).

WHY MEDIAN/MAD INSTEAD OF MEAN/STD (this is the key design decision here):

Day 1 surfaced a real finding: on 2011-12-09, a single order (80,995 units
of "PAPER CRAFT, LITTLE BIRDIE") made up 85% of that day's revenue. A
mean/std z-score detector is built on exactly the statistic that single
event distorts — one extreme day inflates the mean and std of every
rolling window it sits inside, which either (a) makes that day look less
anomalous than it is, or (b) makes every OTHER day look anomalous by
comparison, depending on window position. Median and MAD (median absolute
deviation) are robust to this: a single extreme value barely moves the
median of a 300-point series. This is the standard fix for exactly this
failure mode, not an arbitrary choice.

MAD is scaled by 1.4826 (the standard constant that makes MAD comparable
to a normal distribution's std, so the same z-score thresholds apply).

THRESHOLD: |robust z-score| >= 3.5 flags an anomaly. This is intentionally
a bit more conservative than the textbook 3.0, because this dataset is
already fairly volatile day-to-day (real retail data, not smoothed), and
3.0 produced too many flags on inspection (see validate() below) —
several of which were unremarkable weekday-to-weekend transitions, not
genuine business events.

CONCENTRATION RISK is a separate, rule-based flag (not statistical): if
one line item is >= 50% of a day's revenue, flag it regardless of whether
the day's TOTAL revenue is statistically anomalous. This directly targets
the bulk-order pattern rather than hoping the statistical detector catches
it as a side effect — the Dec 9 case is exactly this: total revenue z-score
alone doesn't reliably distinguish "normal busy day" from "one whale
order", but a direct revenue-share check does.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.history_cache import compute_history  # noqa: E402
from pipeline.metrics import get_connection  # noqa: E402

ROBUST_Z_THRESHOLD = 3.5
CONCENTRATION_THRESHOLD_PCT = 50.0
MAD_SCALE = 1.4826  # makes MAD comparable to normal-distribution std
MIN_HISTORY_DAYS = 14  # don't flag anomalies until we have enough baseline


def build_history(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Core daily metrics for every date in product_sales, recomputed from scratch.
    This is the baseline the robust z-scores are computed against; the pipeline
    itself uses the cached ``history_cache.get_history`` (same result, incremental).
    """
    return compute_history(con)


def robust_zscore(value: float, series: pd.Series) -> float:
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        # Degenerate case: no spread in the baseline. Avoid divide-by-zero;
        # treat any deviation from the median as maximally anomalous only
        # if it's genuinely different, otherwise zero.
        return 0.0 if value == median else float("inf")
    return (value - median) / (mad * MAD_SCALE)


def median_order_value(con: duckdb.DuckDBPyConnection, target_date: date) -> float:
    """
    Per-order (not per-line-item) revenue, median across orders that day.
    Reported alongside mean AOV (from Day 1's metrics.py) specifically
    because the Dec 9 bulk order showed mean AOV can be wildly misleading
    on days with one extreme order — median is far more representative
    of a "typical" order that day.
    """
    row = con.execute("""
        SELECT median(order_revenue) FROM (
            SELECT InvoiceNo, SUM(Quantity * UnitPrice) AS order_revenue
            FROM product_sales
            WHERE make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) = ?
            GROUP BY InvoiceNo
        )
    """, [target_date]).fetchone()
    return round(row[0], 2) if row and row[0] is not None else 0.0


def concentration_risk(con: duckdb.DuckDBPyConnection, target_date: date) -> dict:
    """
    Flags when a single line item accounts for an outsized share of a
    day's revenue — the direct, rule-based counterpart to the Dec 9 bulk
    order finding from Day 1.
    """
    rows = con.execute("""
        SELECT Description, SUM(Quantity * UnitPrice) AS line_rev
        FROM product_sales
        WHERE make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) = ?
        GROUP BY Description
        ORDER BY line_rev DESC
        LIMIT 1
    """, [target_date]).fetchone()

    day_total = con.execute("""
        SELECT SUM(Quantity * UnitPrice)
        FROM product_sales
        WHERE make_date(CAST(year AS INT), CAST(month AS INT), CAST(day AS INT)) = ?
    """, [target_date]).fetchone()[0]

    if not rows or not day_total:
        return {"flagged": False}

    top_product, top_revenue = rows
    pct_of_day = round(top_revenue / day_total * 100, 1)
    flagged = pct_of_day >= CONCENTRATION_THRESHOLD_PCT

    return {
        "flagged": flagged,
        "product": top_product,
        "revenue": round(float(top_revenue), 2),
        "pct_of_day_revenue": pct_of_day,
    }


def detect(con: duckdb.DuckDBPyConnection, target_date: date, history: pd.DataFrame) -> dict:
    """
    Runs the full Day 2 detection suite for one date: statistical
    anomalies (revenue, order_count) via robust z-score, concentration
    risk, return-rate spikes, and median order value.
    """
    prior = history[history["date"] < target_date]
    today_row = history[history["date"] == target_date]

    if today_row.empty:
        return {"date": str(target_date), "has_data": False}

    today = today_row.iloc[0]
    result = {
        "date": str(target_date),
        "has_data": True,
        "median_order_value": median_order_value(con, target_date),
        "concentration_risk": concentration_risk(con, target_date),
        "statistical_anomalies": [],
        "insufficient_history": len(prior) < MIN_HISTORY_DAYS,
    }

    if len(prior) >= MIN_HISTORY_DAYS:
        for metric in ["revenue", "order_count"]:
            z = robust_zscore(today[metric], prior[metric])
            if abs(z) >= ROBUST_Z_THRESHOLD:
                result["statistical_anomalies"].append({
                    "metric": metric,
                    "value": float(today[metric]),
                    "baseline_median": float(prior[metric].median()),
                    "robust_zscore": round(float(z), 2),
                    "direction": "above" if z > 0 else "below",
                })

        # Return-rate spike: cancellations as a share of orders, vs. baseline.
        prior_return_rate = (prior["return_count"] / prior["order_count"].replace(0, np.nan)).median()
        today_return_rate = today["return_count"] / today["order_count"] if today["order_count"] else 0
        if prior_return_rate and today_return_rate >= prior_return_rate * 3 and today["return_count"] >= 3:
            result["statistical_anomalies"].append({
                "metric": "return_rate",
                "value": round(float(today_return_rate), 3),
                "baseline_median": round(float(prior_return_rate), 3),
                "note": "return rate at least 3x the trailing median",
            })

    return result


def validate_against_known_events(con: duckdb.DuckDBPyConnection, history: pd.DataFrame):
    """
    Sanity-check the detector against events we already know about from
    Day 1, rather than trusting it blindly. Prints results for manual
    review — this is a validation script, not a unit test suite (that
    comes in Day 4).
    """
    print("=== Validation: 2011-12-09 (known bulk order day) ===")
    result = detect(con, date(2011, 12, 9), history)
    print(json.dumps(result, indent=2, default=str))

    print("\n=== Full-year scan: how many days get flagged? ===")
    flagged_dates = []
    for d in history["date"]:
        r = detect(con, d, history)
        if r.get("statistical_anomalies") or r.get("concentration_risk", {}).get("flagged"):
            flagged_dates.append((d, r))

    print(f"{len(flagged_dates)} / {len(history)} days flagged "
          f"({len(flagged_dates)/len(history)*100:.1f}%)")
    for d, r in flagged_dates:
        tags = [a["metric"] for a in r["statistical_anomalies"]]
        if r["concentration_risk"]["flagged"]:
            tags.append("concentration_risk")
        print(f"  {d}: {tags}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    con = get_connection(root / "data" / "partitioned")
    history = build_history(con)
    print(f"Built history: {len(history)} days\n")
    validate_against_known_events(con, history)
