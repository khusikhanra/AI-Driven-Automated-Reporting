"""
Does the EXISTING Day 1-3 pipeline consume generated data correctly, and does the
Day 2 detector actually catch the events the generator injects (ground truth)?
"""

import json
import time
from datetime import date

import pandas as pd
import pytest

from pipeline.anomaly_detection import detect
from pipeline.history_cache import _compute_row, get_history
from pipeline.metrics import compute_daily_metrics, get_connection
from pipeline.run_report import run as run_report
from pipeline.synthetic_generator import advance_one_day, latest_partitioned_date
from tests.conftest import HIST_END, ROOT, build_sandbox

PLAN = [  # (seed, event) applied to consecutive trading days from 2011-12-11
    (1, "none"), (2, "bulk_order"), (3, "return_spike"), (4, "demand_spike"),
]


@pytest.fixture(scope="module")
def full(real_root, tmp_path_factory):
    sb = build_sandbox(real_root, tmp_path_factory.mktemp("integration"), full=True)
    out = []
    for seed, ev in PLAN:
        res = advance_one_day(sb, seed=seed, event=ev)
        out.append((res, json.loads((sb / res["files"][-1]).read_text())))
    return sb, out


@pytest.fixture(autouse=True)
def no_live_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_tests_never_pollute_the_real_project_data(real_root):
    assert latest_partitioned_date(ROOT / "data/partitioned") == HIST_END


def test_existing_duckdb_views_read_real_and_synthetic_together(full):
    sb, out = full
    con = get_connection(sb / "data/partitioned")
    n = con.execute("select count(distinct make_date(year::int, month::int, day::int)) from product_sales").fetchone()[0]
    assert n == 305 + len(PLAN)
    types = dict(con.execute("select column_name, column_type from (describe select * from product_sales)").fetchall())
    assert types["InvoiceDate"].startswith("TIMESTAMP") and types["CustomerID"] == "DOUBLE"
    assert types["Quantity"] == "BIGINT" and types["InvoiceNo"] == "VARCHAR"


# Pandas .sum() and DuckDB SQL SUM() add the same floats in a different order,
# so large totals can land a single cent apart (a float-precision artifact, not
# a logic bug). abs=0.02 absorbs that; anything larger would be a real mismatch.
REV_TOL = 0.02


def test_metrics_layer_matches_generator_ground_truth(full):
    sb, out = full
    con = get_connection(sb / "data/partitioned")
    for res, man in out:
        m = compute_daily_metrics(con, date.fromisoformat(res["date"]))
        assert m["has_data"]
        assert m["revenue"] == pytest.approx(man["revenue"], abs=REV_TOL)
        assert m["order_count"] == man["order_count"] and m["return_count"] == man["return_count"]


def test_week_over_week_bridges_real_and_synthetic(full):
    sb, out = full
    con = get_connection(sb / "data/partitioned")
    m = compute_daily_metrics(con, date(2011, 12, 12))          # synthetic vs real 2011-12-05
    assert m["comparisons"]["wow"]["reference_date"] == "2011-12-05"
    assert m["comparisons"]["wow"]["reference_available"] is True


def test_history_cache_extends_incrementally(full):
    sb, out = full
    con = get_connection(sb / "data/partitioned")
    cache = sb / "data/history_cache/history.parquet"
    assert len(pd.read_parquet(cache)) == 305
    t0 = time.time()
    h = get_history(con, cache)
    assert len(h) == 305 + len(PLAN) and time.time() - t0 < 10     # computed only the 4 new days
    for _, man in out:
        row = h[h["date"] == date.fromisoformat(man["date"])].iloc[0]
        assert row["revenue"] == pytest.approx(man["revenue"], abs=REV_TOL)
    assert h["revenue"].dtype == "float64" and h["order_count"].dtype == "int64"


def test_detector_flags_each_injected_event(full):
    sb, out = full
    con = get_connection(sb / "data/partitioned")
    h = get_history(con, sb / "data/history_cache/history.parquet")
    expected = {"bulk_order": "concentration_risk", "return_spike": "return_rate", "demand_spike": "revenue"}
    for res, man in out:
        d = detect(con, date.fromisoformat(res["date"]), h)
        flags = {a["metric"] for a in d["statistical_anomalies"]}
        if d["concentration_risk"]["flagged"]:
            flags.add("concentration_risk")
        ev = man["event"]["name"]
        if ev in expected:
            assert expected[ev] in flags, (ev, flags)


def test_full_report_for_every_generated_day(full):
    sb, out = full
    for res, man in out:
        result = run_report(date.fromisoformat(res["date"]), sb)
        assert result["status"] == "ok" and result["context"]["data_source"] == "synthetic"
        html = (sb / "reports" / f"report_{res['date']}.html").read_text(encoding="utf-8")
        assert "Simulated data" in html and f"seed {man['seed']}" in html
        assert man["event"]["name"] not in html.replace("_", " ") or man["event"]["name"] == "none"  # ground truth not leaked
        if man["event"]["name"] == "bulk_order":
            assert "Concentration risk" in html
        if man["event"]["name"] == "return_spike":
            assert "Return rate spike" in html


def test_real_historical_report_is_not_labelled_simulated(full):
    sb, _ = full
    assert run_report(HIST_END, sb)["context"]["data_source"] == "historical"
    assert "Simulated data" not in (sb / "reports" / f"report_{HIST_END}.html").read_text(encoding="utf-8")


# ------------------------------------------------------------------ ground-truth detection rates

def _flags(sb, base, D):
    con = get_connection(sb / "data/partitioned")
    hist = pd.concat([base, pd.DataFrame([_compute_row(con, D)])], ignore_index=True)
    d = detect(con, D, hist)
    flags = {a["metric"] for a in d["statistical_anomalies"]}
    if d["concentration_risk"]["flagged"]:
        flags.add("concentration_risk")
    return flags


@pytest.fixture(scope="module")
def trial_env(real_root, tmp_path_factory):
    sb = build_sandbox(real_root, tmp_path_factory.mktemp("trials"))
    base = pd.read_parquet(real_root / "data/history_cache/history.parquet")
    base["date"] = pd.to_datetime(base["date"]).dt.date
    return sb, base[base["date"] <= HIST_END]


D = date(2011, 12, 13)


@pytest.mark.parametrize("event,signal", [("bulk_order", "concentration_risk"),
                                          ("demand_spike", "revenue"),
                                          ("return_spike", "return_rate")])
def test_detector_catches_injected_events_across_seeds(trial_env, event, signal):
    sb, base = trial_env
    hits = 0
    for seed in range(1, 9):
        advance_one_day(sb, seed=seed, event=event, target_date=D, overwrite=True)
        hits += signal in _flags(sb, base, D)
    assert hits == 8, f"{event}: detected {hits}/8"


def test_false_positive_rate_on_normal_days_is_low(trial_env):
    sb, base = trial_env
    flagged = 0
    for seed in range(1, 13):
        advance_one_day(sb, seed=seed, event="none", target_date=D, overwrite=True)
        flagged += bool(_flags(sb, base, D))
    assert flagged <= 3, f"{flagged}/12 normal days flagged (historical rate is ~9%)"


def test_known_blind_spot_demand_drop_is_not_detected(trial_env):
    """Documents a Day 2 detector limitation (see generator docstring / final report).
    If the detector later gains low-side detection, THIS test should flip to assert detection."""
    sb, base = trial_env
    for seed in range(1, 7):
        res = advance_one_day(sb, seed=seed, event="demand_drop", target_date=D, overwrite=True)
        assert res["expected_detectable"] is False
        assert _flags(sb, base, D) == set()
