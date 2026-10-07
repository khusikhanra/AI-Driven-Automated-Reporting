"""
Tests for the synthetic generator (Synthetic Profile v2.0). Seeds are fixed everywhere, so every
statistical assertion is deterministic (no flaky tests): the tolerances below
were set from measured behaviour and are deliberately wide enough to be
meaningful without being brittle.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import pipeline.synthetic_generator as sg
from config.data_rules import is_non_product_code
from pipeline.synthetic_generator import (EVENTS, GeneratorError, advance_one_day, choose_event,
                                          expected_orders, is_trading_day, next_trading_day)
from tests.conftest import HIST_END, ROOT

TUE = date(2011, 12, 13)


# ----------------------------------------------------------------------------- helpers

def load_day(sb, res):
    ps = pd.read_parquet(sb / res["files"][0])
    cx_files = [f for f in res["files"] if "/cancellations/" in f]
    cx = pd.read_parquet(sb / cx_files[0]) if cx_files else pd.DataFrame(columns=ps.columns)
    manifest = json.loads((sb / res["files"][-1]).read_text())
    return ps, cx, manifest


def strip(manifest):
    return {k: v for k, v in manifest.items() if k != "generated_at"}


# ----------------------------------------------------------------------------- schema

class TestSchema:
    def test_product_sales_schema_identical_to_historical(self, sandbox):
        sb = sandbox()
        ps, _, _ = load_day(sb, advance_one_day(sb, seed=1, event="none"))
        hist = pd.read_parquet(ROOT / "data/cleaned/product_sales.parquet")
        assert list(ps.columns) == list(hist.columns)
        assert (ps.dtypes == hist.dtypes).all(), (ps.dtypes, hist.dtypes)

    def test_cancellations_schema_and_signs(self, sandbox):
        sb = sandbox()
        _, cx, _ = load_day(sb, advance_one_day(sb, seed=1, event="return_spike"))
        hist = pd.read_parquet(ROOT / "data/cleaned/cancellations.parquet")
        assert list(cx.columns) == list(hist.columns) and (cx.dtypes == hist.dtypes).all()
        assert len(cx) > 0
        assert cx["InvoiceNo"].str.startswith("C").all()
        assert (cx["Quantity"] < 0).all() and (cx["UnitPrice"] > 0).all()
        assert cx["is_cancellation"].all()
        assert np.allclose(cx["line_revenue"], cx["Quantity"] * cx["UnitPrice"])

    def test_product_sales_values_are_valid(self, sandbox):
        sb = sandbox()
        ps, _, _ = load_day(sb, advance_one_day(sb, seed=2, event="none"))
        assert (ps["Quantity"] > 0).all() and (ps["UnitPrice"] > 0).all()
        for col in ("InvoiceNo", "StockCode", "Description", "Country", "InvoiceDate"):
            assert ps[col].notna().all(), col
        assert not ps["is_cancellation"].any() and not ps["is_non_product"].any()
        assert not ps["StockCode"].map(is_non_product_code).any()   # postage was cleaned out
        assert np.allclose(ps["line_revenue"], ps["Quantity"] * ps["UnitPrice"])
        assert ps["InvoiceNo"].str.fullmatch(r"\d{6,}").all()

    def test_invoice_integrity_and_identifiers(self, sandbox, real_root):
        sb = sandbox()
        res = advance_one_day(sb, seed=3, event="none")
        ps, cx, _ = load_day(sb, res)
        for col in ("InvoiceDate", "Country"):                      # one value per invoice
            assert (ps.groupby("InvoiceNo")[col].nunique() == 1).all()
        assert (ps.groupby("InvoiceNo")["CustomerID"].nunique(dropna=False) == 1).all()
        hist_max = json.loads((real_root / "data/profile/profile.json").read_text())["max_invoice_no"]
        sale_nums = ps["InvoiceNo"].astype(int).unique()
        cancel_nums = cx["InvoiceNo"].str[1:].astype(int).unique()
        assert sale_nums.min() > hist_max
        assert not set(sale_nums) & set(cancel_nums)
        assert ps.sort_values("InvoiceDate", kind="stable")["InvoiceNo"].astype(int).is_monotonic_increasing  # ties keep row order

    def test_timestamps_fall_on_target_day_within_business_hours(self, sandbox):
        sb = sandbox()
        res = advance_one_day(sb, seed=4, event="none")
        ps, cx, _ = load_day(sb, res)
        for df in (ps, cx):
            assert (df["InvoiceDate"].dt.date == date.fromisoformat(res["date"])).all()
            assert df["InvoiceDate"].dt.hour.between(6, 20).all()

    def test_values_come_from_the_learned_universe(self, sandbox, real_root):
        prof = json.loads((real_root / "data/profile/profile.json").read_text())
        sb = sandbox()
        ps, cx, _ = load_day(sb, advance_one_day(sb, seed=5, event="none"))
        allrows = pd.concat([ps, cx])
        assert set(allrows["StockCode"]) <= {p["code"] for p in prof["products"]}
        assert set(allrows["Country"]) <= set(prof["countries"])
        known = allrows["CustomerID"].dropna()
        assert (known.isin([c["id"] for c in prof["customers"]]) | (known > prof["max_customer_id"])).all()

    def test_partition_layout_and_manifest(self, sandbox):
        sb = sandbox()
        res = advance_one_day(sb, seed=6, event="none")
        assert res["files"][0] == "data/partitioned/product_sales/year=2011/month=12/day=11/orders.parquet"
        assert res["files"][-1] == "data/synthetic_manifest/date=2011-12-11.json"
        assert all((sb / f).exists() for f in res["files"])
        man = json.loads((sb / res["files"][-1]).read_text())
        for key in ("is_synthetic", "seed", "event", "profile_fingerprint", "historical_end_date",
                    "revenue", "order_count", "return_count", "clean_report"):
            assert key in man
        assert man["is_synthetic"] is True and man["clean_report"]["reconciliation_ok"] is True


# ----------------------------------------------------------------------------- dates

class TestDates:
    def test_first_synthetic_day_skips_closed_saturday(self, sandbox):
        res = advance_one_day(sandbox(), seed=1, event="none")
        assert res["date"] == "2011-12-11"            # Fri 12-09 is the last real day; Sat 12-10 is closed
        assert res["skipped_closed_days"] == ["2011-12-10"]
        assert date.fromisoformat(res["date"]) > HIST_END

    def test_consecutive_runs_advance_one_trading_day_each(self, sandbox):
        sb = sandbox()
        dates = [advance_one_day(sb, seed=i, event="none")["date"] for i in range(5)]
        assert dates == ["2011-12-11", "2011-12-12", "2011-12-13", "2011-12-14", "2011-12-15"]

    def test_saturday_gap_between_friday_and_sunday(self, sandbox):
        sb = sandbox()
        got = [advance_one_day(sb, seed=i, event="none")["date"] for i in range(8)]
        assert "2011-12-17" not in got and "2011-12-16" in got and "2011-12-18" in got

    def test_christmas_closure_is_skipped(self, model):
        d, skipped = next_trading_day(model, date(2011, 12, 23))
        assert d == date(2012, 1, 4)
        assert skipped[0] == "2011-12-24" and skipped[-1] == "2012-01-03" and len(skipped) == 11

    def test_trading_calendar_rules(self, model):
        assert not is_trading_day(model, date(2011, 12, 10))   # Saturday
        assert is_trading_day(model, date(2011, 12, 11))       # Sunday trades
        assert not is_trading_day(model, date(2011, 12, 25))
        assert is_trading_day(model, date(2012, 1, 4))

    @pytest.mark.parametrize("bad", [HIST_END, date(2011, 11, 1), date(2010, 12, 1)])
    def test_refuses_to_touch_historical_dates(self, sandbox, bad):
        sb = sandbox()
        with pytest.raises(GeneratorError, match="historical end"):
            advance_one_day(sb, seed=1, target_date=bad)

    def test_refuses_closed_day(self, sandbox):
        with pytest.raises(GeneratorError, match="non-trading"):
            advance_one_day(sandbox(), seed=1, target_date=date(2011, 12, 17))

    def test_refuses_dates_too_far_ahead(self, sandbox):
        with pytest.raises(GeneratorError, match="more than"):
            advance_one_day(sandbox(), seed=1, target_date=date(2012, 2, 1))

    def test_existing_day_requires_overwrite(self, sandbox):
        sb = sandbox()
        advance_one_day(sb, seed=1, event="none", target_date=TUE)
        with pytest.raises(GeneratorError, match="already exists"):
            advance_one_day(sb, seed=1, event="none", target_date=TUE)
        res = advance_one_day(sb, seed=99, event="bulk_order", target_date=TUE, overwrite=True)
        assert json.loads((sb / res["files"][-1]).read_text())["seed"] == 99

    def test_overwrite_leaves_no_stale_partitions(self, sandbox):
        sb = sandbox()
        advance_one_day(sb, seed=1, event="return_spike", target_date=TUE)
        res = advance_one_day(sb, seed=1, event="none", target_date=TUE, overwrite=True)
        ps, cx, man = load_day(sb, res)
        assert len(cx) == man["cancellation_rows"] and len(ps) == man["product_sales_rows"]

    def test_invoice_numbers_unique_across_days(self, sandbox):
        sb = sandbox()
        seen = set()
        for i in range(3):
            ps, cx, _ = load_day(sb, advance_one_day(sb, seed=i, event="none"))
            nums = set(ps["InvoiceNo"]) | set(cx["InvoiceNo"].str[1:])
            assert not nums & seen
            seen |= nums


# ----------------------------------------------------------------------------- reproducibility

class TestReproducibility:
    def test_same_seed_same_date_is_identical(self, sandbox):
        a, b = sandbox(), sandbox()
        ra, rb = advance_one_day(a, seed=42, event="none"), advance_one_day(b, seed=42, event="none")
        (psa, cxa, ma), (psb, cxb, mb) = load_day(a, ra), load_day(b, rb)
        pd.testing.assert_frame_equal(psa, psb)
        pd.testing.assert_frame_equal(cxa, cxb)
        assert strip(ma) == strip(mb)

    @pytest.mark.parametrize("event", sorted(EVENTS))
    def test_every_event_is_reproducible(self, sandbox, event):
        a, b = sandbox(), sandbox()
        ra, rb = advance_one_day(a, seed=7, event=event), advance_one_day(b, seed=7, event=event)
        pd.testing.assert_frame_equal(load_day(a, ra)[0], load_day(b, rb)[0])

    def test_different_seeds_differ(self, sandbox):
        a, b = sandbox(), sandbox()
        psa = load_day(a, advance_one_day(a, seed=1, event="none"))[0]
        psb = load_day(b, advance_one_day(b, seed=2, event="none"))[0]
        assert not (len(psa) == len(psb) and psa["line_revenue"].sum() == psb["line_revenue"].sum())

    def test_same_seed_different_dates_differ(self, sandbox):
        sb = sandbox()
        r1, r2 = advance_one_day(sb, seed=5, event="none"), advance_one_day(sb, seed=5, event="none")
        m1, m2 = load_day(sb, r1)[2], load_day(sb, r2)[2]
        assert r1["date"] != r2["date"] and m1["revenue"] != m2["revenue"]

    def test_live_mode_reports_seed_and_can_be_replayed(self, sandbox):
        a, b, c = sandbox(), sandbox(), sandbox()
        live1, live2 = advance_one_day(a), advance_one_day(b)     # seed=None, event=None
        assert live1["seed"] != live2["seed"]                     # live runs vary
        replay = advance_one_day(c, seed=live1["seed"])           # replay from the logged seed only
        assert replay["event"] == live1["event"]                  # even the random event choice replays
        pd.testing.assert_frame_equal(load_day(a, live1)[0], load_day(c, replay)[0])

    def test_event_choice_is_deterministic(self):
        assert [choose_event(5, TUE) for _ in range(3)] == [choose_event(5, TUE)] * 3

    def test_baseline_volume_is_independent_of_event_choice(self, sandbox):
        base = {}
        for ev in ("none", "bulk_order", "demand_spike", "promo_uplift"):
            sb = sandbox()
            man = load_day(sb, advance_one_day(sb, seed=11, event=ev))[2]
            base[ev] = man["event"]["baseline_orders"]
        assert len(set(base.values())) == 1, base


# ----------------------------------------------------------------------------- normal generation

@pytest.fixture(scope="module")
def days(real_root, tmp_path_factory):
    """30 normal Tuesdays, one per seed (invoice numbers repeat across seeds by design,
    so anything per-invoice must be computed per day, never pooled)."""
    from tests.conftest import build_sandbox
    sb = build_sandbox(real_root, tmp_path_factory.mktemp("normal"))
    out = []
    for seed in range(1, 31):
        res = advance_one_day(sb, seed=seed, event="none", target_date=TUE, overwrite=True)
        out.append(load_day(sb, res))
    return out


class TestNormalGeneration:
    def test_daily_volume_matches_the_learned_expectation(self, days, model):
        mean_orders = np.mean([m["order_count"] for _, _, m in days])
        assert 0.75 * expected_orders(model, TUE) <= mean_orders <= 1.35 * expected_orders(model, TUE)

    def test_daily_revenue_is_in_a_realistic_range(self, days, model):
        med = np.median([m["revenue"] for _, _, m in days])
        assert 0.5 * model.scale["revenue_median"] <= med <= 2.5 * model.scale["revenue_median"]

    def test_cancellation_rate_is_realistic(self, days):
        rates = [m["return_count"] / m["order_count"] for _, _, m in days]
        assert 0.08 <= np.median(rates) <= 0.30              # historical median 0.163

    def test_guest_and_basket_shape(self, days):
        ps = pd.concat([d[0] for d in days])
        assert 0.05 <= ps["CustomerID"].isna().mean() <= 0.40   # historical ~0.25 of lines
        sizes = np.concatenate([d[0].groupby("InvoiceNo").size().to_numpy() for d in days])
        assert 8 <= np.median(sizes) <= 25                          # historical median 15

    def test_postage_lines_are_generated_then_cleaned_out(self, days):
        assert sum(m["clean_report"]["non_product_adjustment_rows"] for _, _, m in days) > 0
        assert all(m["clean_report"]["reconciliation_ok"] for _, _, m in days)

    def test_normal_days_have_no_freak_lines(self, days, model):
        for ps, _, m in days:
            assert ps["Quantity"].max() <= model.scale["qty_cap"]
            assert ps["line_revenue"].max() / ps["line_revenue"].sum() < 0.5

    def test_weekday_pattern_sunday_lighter_than_weekdays(self, sandbox, model):
        assert expected_orders(model, date(2011, 12, 11)) < expected_orders(model, date(2011, 12, 15))


# ----------------------------------------------------------------------------- events

class TestEvents:
    def _one(self, sandbox, event, seed=3):
        sb = sandbox()
        res = advance_one_day(sb, seed=seed, event=event, target_date=TUE)
        return (*load_day(sb, res), res)

    def test_none_event_has_no_parameters(self, sandbox):
        _, _, man, res = self._one(sandbox, "none")
        assert man["event"]["name"] == "none" and man["event"]["params"] == {}
        assert res["expected_detectable"] is False

    def test_bulk_order_dominates_revenue(self, sandbox):
        ps, _, man, res = self._one(sandbox, "bulk_order")
        top = ps["line_revenue"].max() / ps["line_revenue"].sum()
        assert top >= 0.54 and man["event"]["params"]["quantity"] > 435
        assert ps.groupby("InvoiceNo").size()[ps.loc[ps["line_revenue"].idxmax(), "InvoiceNo"]] == 1
        assert res["expected_detectable"] and man["event"]["expected_signal"] == "concentration_risk"

    def test_demand_spike_reaches_target_revenue(self, sandbox, model):
        _, _, man, _ = self._one(sandbox, "demand_spike")
        p = man["event"]["params"]
        assert man["revenue"] >= p["target_revenue"]
        assert p["severity_robust_sigma"] >= 4.5
        assert man["revenue"] >= model.scale["revenue_median"] + 4.5 * model.scale["revenue_sigma"]

    def test_return_spike_lifts_cancellation_rate(self, sandbox):
        _, cx, man, _ = self._one(sandbox, "return_spike")
        assert man["return_count"] >= 3 and man["return_count"] / man["order_count"] >= 0.5

    def test_demand_drop_targets_its_physical_floor(self, sandbox, model):
        """demand_drop is calibrated to the MOST SEVERE drop physically possible
        (see synthetic_generator.py module docstring for the proof this is still
        provably undetectable at the current threshold -- that is the point)."""
        _, _, man, res = self._one(sandbox, "demand_drop")
        p = man["event"]["params"]
        assert man["order_count"] == p["target_orders"]
        assert p["target_orders"] <= max(1, round(model.scale["orders_median"] - 4.5 * model.scale["orders_sigma"]))
        assert 4.5 <= p["severity_robust_sigma"] <= 6.0
        assert res["expected_detectable"] is False

    def test_demand_drop_is_provably_below_the_detection_threshold(self, sandbox, model):
        """Formal check of the claim in the module docstring: even order_count=1,
        the most extreme drop physically possible, cannot reach the 3.5 threshold
        given this dataset's own median/MAD. If this ever starts failing, either
        the real data changed or the detector threshold changed -- both are
        reasons to revisit whether demand_drop is still undetectable."""
        max_achievable_z = (model.scale["orders_median"] - 1) / model.scale["orders_sigma"]
        assert max_achievable_z < 3.5

    def test_promo_uplift_raises_volume_moderately(self, sandbox):
        _, _, man, _ = self._one(sandbox, "promo_uplift")
        ratio = man["order_count"] / man["event"]["baseline_orders"]
        assert 1.2 <= ratio <= 1.7

    def test_unknown_event_rejected(self, sandbox):
        with pytest.raises(GeneratorError, match="Unknown event"):
            advance_one_day(sandbox(), seed=1, event="meteor_strike")

    def test_random_event_rate_and_coverage(self):
        picks = [choose_event(s, TUE) for s in range(600)]
        rate = 1 - picks.count("none") / len(picks)
        assert 0.08 <= rate <= 0.17                            # target 0.12
        assert {p for p in picks if p != "none"} == {n for n in EVENTS if n != "none"}

    def test_event_probability_bounds(self):
        assert all(choose_event(s, TUE, 0.0) == "none" for s in range(50))
        assert all(choose_event(s, TUE, 1.0) != "none" for s in range(50))

    def test_ground_truth_catalogue_is_consistent(self):
        assert EVENTS["bulk_order"]["signal"] == "concentration_risk"
        assert EVENTS["demand_drop"]["detectable"] is False        # documented blind spot
        assert abs(sum(e["weight"] for e in EVENTS.values()) - 1.0) < 1e-9


# ----------------------------------------------------------------------------- failures / edge cases

class TestFailures:
    def test_missing_profile_and_cleaned_data(self, tmp_path):
        with pytest.raises(GeneratorError, match="No profile"):
            advance_one_day(tmp_path, seed=1)

    def test_corrupt_profile(self, sandbox):
        sb = sandbox()
        (sb / "data/profile/profile.json").write_text("{not json")
        with pytest.raises(GeneratorError, match="unreadable"):
            advance_one_day(sb, seed=1)

    def test_profile_version_mismatch(self, sandbox):
        sb = sandbox()
        p = sb / "data/profile/profile.json"
        d = json.loads(p.read_text())
        d["profile_version"] = "1.0"          # a pre-v2 profile must be rejected, never half-read
        p.write_text(json.dumps(d))
        with pytest.raises(GeneratorError, match="version"):
            advance_one_day(sb, seed=1)

    def test_tampered_profile_is_rejected(self, sandbox):
        sb = sandbox()
        p = sb / "data/profile/profile.json"
        d = json.loads(p.read_text())
        d["scale"]["orders_median"] += 1       # content changed, fingerprint not updated
        p.write_text(json.dumps(d))
        with pytest.raises(GeneratorError, match="fingerprint"):
            advance_one_day(sb, seed=1)

    def test_profile_is_rebuilt_from_cleaned_data_when_absent(self, sandbox):
        import shutil
        sb = sandbox()
        shutil.rmtree(sb / "data/profile")
        assert advance_one_day(sb, seed=1, event="none")["status"] == "advanced"
        assert (sb / "data/profile/profile.json").exists()

    def test_no_partitioned_data(self, sandbox):
        import shutil
        sb = sandbox()
        shutil.rmtree(sb / "data/partitioned")
        with pytest.raises(GeneratorError, match="No partitioned data"):
            advance_one_day(sb, seed=1)

    def test_incomplete_history_is_refused(self, sandbox):
        sb = sandbox(last_day=date(2011, 12, 8))
        with pytest.raises(GeneratorError, match="incomplete"):
            advance_one_day(sb, seed=1)

    @pytest.mark.parametrize("bad", [-1, "abc", 1.5, True, 2 ** 63])
    def test_invalid_seed_rejected(self, sandbox, bad):
        with pytest.raises(GeneratorError, match="seed"):
            advance_one_day(sandbox(), seed=bad)

    @pytest.mark.parametrize("p", [-0.1, 1.5])
    def test_invalid_event_probability(self, sandbox, p):
        with pytest.raises(GeneratorError, match="event_probability"):
            advance_one_day(sandbox(), seed=1, event_probability=p)

    def test_failures_leave_no_side_effects(self, sandbox):
        sb = sandbox()
        before = sorted(str(p) for p in sb.rglob("*"))
        for kwargs in ({"target_date": date(2011, 12, 17)}, {"event": "nope"}, {"seed": -5}):
            with pytest.raises(GeneratorError):
                advance_one_day(sb, **{"seed": 1, **kwargs})
        assert sorted(str(p) for p in sb.rglob("*")) == before

    def test_cleaning_reconciliation_failure_writes_nothing(self, sandbox, monkeypatch):
        sb = sandbox()
        real_clean = sg.clean

        def broken(df):
            out = real_clean(df)
            out["report"]["reconciliation_ok"] = False
            return out
        monkeypatch.setattr(sg, "clean", broken)
        before = sorted(str(p) for p in sb.rglob("*"))
        with pytest.raises(GeneratorError, match="reconciliation"):
            advance_one_day(sb, seed=1, event="none")
        assert sorted(str(p) for p in sb.rglob("*")) == before

    def test_identifier_block_overflow_is_caught(self, sandbox, monkeypatch):
        monkeypatch.setattr(sg, "INVOICE_BLOCK", 10)
        with pytest.raises(GeneratorError, match="identifier block"):
            advance_one_day(sandbox(), seed=1, event="none")

    def test_tiny_expected_volume_still_yields_a_valid_day(self, sandbox):
        sb = sandbox()
        for seed in range(15):       # demand_drop can shrink volume to a handful of orders
            res = advance_one_day(sb, seed=seed, event="demand_drop", target_date=TUE, overwrite=True)
            ps, _, man = load_day(sb, res)
            assert man["order_count"] >= 1 and len(ps) >= 1 and man["clean_report"]["reconciliation_ok"]


# ----------------------------------------------------------------------------- consistency & manifest

class TestConsistency:
    """Cross-checks between a generated day, the v2 profile and its manifest."""

    def test_customer_country_pairs_come_from_the_profile(self, sandbox, real_root):
        prof = json.loads((real_root / "data/profile/profile.json").read_text())
        country_of = {c["id"]: prof["countries"][c["country_idx"]] for c in prof["customers"]}
        sb = sandbox()
        ps, cx, _ = load_day(sb, advance_one_day(sb, seed=8, event="none", target_date=TUE))
        known = pd.concat([ps, cx]).dropna(subset=["CustomerID"])
        existing = known[known["CustomerID"].isin(country_of)]
        assert len(existing) > 0
        assert (existing["CustomerID"].map(country_of) == existing["Country"]).all()
        new = known[~known["CustomerID"].isin(country_of)]
        assert (new["CustomerID"] > prof["max_customer_id"]).all()          # new customers never collide

    def test_quantities_respect_the_profile_cap_on_normal_days(self, sandbox, model):
        sb = sandbox()
        ps, cx, _ = load_day(sb, advance_one_day(sb, seed=9, event="none"))
        assert ps["Quantity"].max() <= model.scale["qty_cap"]
        assert cx["Quantity"].abs().max() <= model.scale["qty_cap"]

    def test_postage_never_reaches_product_sales_but_is_accounted_for(self, sandbox):
        sb = sandbox()
        res = advance_one_day(sb, seed=10, event="none", target_date=TUE)
        ps, _, man = load_day(sb, res)
        assert "POST" not in set(ps["StockCode"])
        assert man["clean_report"]["non_product_adjustment_rows"] > 0
        assert man["clean_report"]["rows_accounted_for"] == man["clean_report"]["input_rows"]

    def test_manifest_has_everything_needed_to_reproduce_the_run(self, sandbox, model):
        sb = sandbox()
        res = advance_one_day(sb, seed=123, event="demand_spike", target_date=TUE)
        man = load_day(sb, res)[2]
        assert man["seed"] == 123 and man["event"]["name"] == "demand_spike" and man["date"] == TUE.isoformat()
        assert man["profile_fingerprint"] == model.profile_fingerprint
        assert man["generator_version"] == sg.GENERATOR_VERSION
        assert man["historical_end_date"] == model.hist_end.isoformat()
        assert man["event"]["params"]["target_revenue"] > 0
        # Replaying from the manifest alone gives the same data.
        other = sandbox()
        res2 = advance_one_day(other, seed=man["seed"], event=man["event"]["name"], target_date=date.fromisoformat(man["date"]))
        pd.testing.assert_frame_equal(load_day(sb, res)[0], load_day(other, res2)[0])

    def test_profile_fingerprint_changes_when_the_profile_changes(self, real_root, tmp_path):
        from pipeline.synthetic_profile import build_profile
        p = build_profile(real_root / "data/cleaned/product_sales.parquet", real_root / "data/cleaned/cancellations.parquet",
                          tmp_path / "p.json")
        again = build_profile(real_root / "data/cleaned/product_sales.parquet", real_root / "data/cleaned/cancellations.parquet",
                              tmp_path / "p2.json")
        assert p["profile_fingerprint"] == again["profile_fingerprint"]            # deterministic
        assert p["profile_version"] == "2.0"

    def test_horizon_boundary(self, sandbox):
        sb = sandbox()
        last_ok = date(2011, 12, 23)          # exactly latest (12-09) + 14 days, a Friday
        assert advance_one_day(sb, seed=1, event="none", target_date=last_ok)["date"] == "2011-12-23"
        with pytest.raises(GeneratorError, match="more than"):
            advance_one_day(sandbox(), seed=1, target_date=date(2012, 1, 4))      # first trading day past the horizon

    def test_non_date_target_is_rejected(self, sandbox):
        with pytest.raises(GeneratorError, match="datetime.date"):
            advance_one_day(sandbox(), seed=1, target_date="2011-12-13")

    @pytest.mark.parametrize("event", sorted(EVENTS))
    def test_every_event_produces_a_valid_reconciled_day(self, sandbox, event):
        sb = sandbox()
        res = advance_one_day(sb, seed=21, event=event, target_date=TUE)
        ps, cx, man = load_day(sb, res)
        assert man["clean_report"]["reconciliation_ok"] and len(ps) > 0
        assert man["event"]["name"] == event and res["expected_detectable"] == EVENTS[event]["detectable"]
        assert not ps["is_cancellation"].any() and (cx.empty or cx["is_cancellation"].all())
        assert np.allclose(ps["line_revenue"], ps["Quantity"] * ps["UnitPrice"])
        assert man["revenue"] == pytest.approx(ps["line_revenue"].sum(), abs=0.01)
