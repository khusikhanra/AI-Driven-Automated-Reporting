"""
synthetic_generator.py - continues the real UCI Online Retail history with
synthetic trading days sampled from the Synthetic Profile v2.0.

Guarantees
----------
* Real history is never touched: only dates after the profile's historical end
  can be generated.
* Deterministic: the same (seed, date, event) always yields byte-identical data.
  A live run (no seed) draws a fresh seed and reports it so it can be replayed.
* Trading calendar comes from the profile (weekdays that traded historically plus
  recurring closures such as the year-end shutdown). Saturday does not trade.
* Generated rows go through the same ``clean()`` as the real history, so
  partitions share one schema; cleaning must reconcile or nothing is written.
* Invoice numbers and new customer ids are unique across days and depend only on
  the target date, never on invocation order.
* A failed run writes nothing (and removes any partial output).

Injected events (ground truth is logged in the manifest, never in the report)
-----------------------------------------------------------------------------
bulk_order    one cheap line worth >1.2x the rest of the day  -> concentration_risk
demand_spike  revenue >= median + 4.5-6 robust sigma           -> revenue anomaly
return_spike  70-100% of sales also get a cancellation         -> return_rate anomaly
demand_drop   volume cut to its physical floor                 -> NOT detectable
promo_uplift  ordinary 1.3-1.6x volume increase                -> not guaranteed

Why demand_drop is undetectable by the current detector: order count is bounded
below by 1 (and revenue by 0) but the detector's z-score threshold (3.5) is on an
unbounded-above scale. The most extreme possible drop therefore reaches only
|z| = (orders_median - 1) / orders_sigma, which is below 3.5 on this dataset
(see ``tests/test_synthetic_generator.py``). That is a property of the data's
dispersion, not a calibration bug; fixing it needs a detector redesign
(asymmetric or log-scale thresholds), which is out of scope here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import paths  # noqa: E402
from pipeline.clean import REQUIRED_COLUMNS, clean  # noqa: E402
from pipeline.partition import write_daily_partitions  # noqa: E402
from pipeline.paths import manifest_path  # noqa: E402,F401  (re-exported for callers)
from pipeline.synthetic_profile import (  # noqa: E402,F401
    GeneratorError,
    SyntheticModel,
    is_trading_day,
    load_or_build_profile,
    next_trading_day,
)

GENERATOR_VERSION = "2.0"

INVOICE_BLOCK = 10_000          # invoice numbers reserved per target date
NEW_CUSTOMER_BLOCK = 1_000      # new customer ids reserved per target date
NEW_CUSTOMER_PROB = 0.03
POSTAGE_PROB = 0.05
POSTAGE_PRICES = np.array([15.0, 18.0, 35.0, 60.0])
POSTAGE_P = np.array([0.20, 0.50, 0.15, 0.15])
MAX_DAYS_AHEAD = 14
DEFAULT_EVENT_PROBABILITY = 0.12
MAX_TOPUP_ORDERS = 5_000
BUSINESS_HOURS = (8, 20)        # [start, end) hours orders are placed in

EVENTS = {
    "none": {"weight": 0.00, "signal": None, "detectable": False},
    "bulk_order": {"weight": 0.25, "signal": "concentration_risk", "detectable": True},
    "demand_spike": {"weight": 0.20, "signal": "revenue", "detectable": True},
    "return_spike": {"weight": 0.25, "signal": "return_rate", "detectable": True},
    "demand_drop": {"weight": 0.15, "signal": None, "detectable": False},
    "promo_uplift": {"weight": 0.15, "signal": None, "detectable": False},
}


@dataclass
class GeneratedDay:
    raw: pd.DataFrame
    info: dict


# ---------------------------------------------------------------------------
# Sampling arrays built from the profile
# ---------------------------------------------------------------------------


def _weights(values) -> np.ndarray:
    """Finite, non-negative weights normalised to sum to 1 (uniform if they sum to 0)."""
    arr = np.clip(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total = arr.sum()
    return arr / total if total > 0 else np.full(arr.shape, 1.0 / len(arr))


class _Sampler:
    """Numpy view of a ``SyntheticModel``: everything the day generator draws from."""

    def __init__(self, model: SyntheticModel):
        self.model = model
        season = model.seasonality
        self.base = float(season["base_orders_per_day"])
        self.sigma = max(0.0, float(season["residual_sigma"]))
        self.dow_factor = {int(k): float(v) for k, v in season["dow_factor"].items()}
        self.month_factor = {int(k): float(v) for k, v in season["month_factor"].items()}

        self.hour_p = _weights(model.hour_weights[BUSINESS_HOURS[0]:BUSINESS_HOURS[1]])
        self.guest_prob = float(np.clip(model.basket["guest_prob"], 0.0, 1.0))
        self.lines_known = self._dist(model.basket["lines_known"])
        self.lines_guest = self._dist(model.basket["lines_guest"])
        self.cancel_lines = self._dist(model.cancellations["lines"])
        self.cancel_rates = np.clip(np.asarray(model.cancellations["rates"], dtype=float), 0.0, 1.0)

        products = model.products
        self.codes = np.array([p["code"] for p in products], dtype=object)
        self.descs = np.array([p["description"] for p in products], dtype=object)
        self.prices = np.array([p["price"] for p in products], dtype=float)
        self.prod_p = _weights([p["weight"] for p in products])
        self.n_prod = len(products)
        samples = [np.asarray(p["qty_samples"], dtype=np.int64) for p in products]
        self.q_len = np.array([len(s) for s in samples], dtype=np.int64)
        self.q_off = np.concatenate([[0], np.cumsum(self.q_len)[:-1]]).astype(np.int64)
        self.q_flat = np.concatenate(samples)
        cheap = np.flatnonzero(self.prices <= 5.0)
        self.cheap_idx = cheap if cheap.size else np.arange(self.n_prod)
        self.cheap_p = _weights(self.prod_p[self.cheap_idx])

        self.countries = model.countries
        self.cust_id = np.array([c["id"] for c in model.customers], dtype=float)
        self.cust_country = np.array([c["country_idx"] for c in model.customers], dtype=int)
        self.cust_cum = np.cumsum(_weights([c["weight"] for c in model.customers]))
        self.guest_country_idx = np.array(model.guest_countries["idx"], dtype=int)
        self.guest_cum = np.cumsum(_weights(model.guest_countries["weight"]))

    @staticmethod
    def _dist(dist: dict) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(dist["values"], dtype=int), _weights(dist["counts"])

    @staticmethod
    def _draw(dist: tuple[np.ndarray, np.ndarray], rng) -> int:
        values, probs = dist
        return int(values[rng.choice(len(values), p=probs)])

    def pick_customer(self, rng) -> tuple[float, str]:
        j = min(int(np.searchsorted(self.cust_cum, rng.random(), side="right")), len(self.cust_id) - 1)
        return float(self.cust_id[j]), self.countries[self.cust_country[j]]

    def pick_guest_country(self, rng) -> str:
        j = min(int(np.searchsorted(self.guest_cum, rng.random(), side="right")), len(self.guest_cum) - 1)
        return self.countries[self.guest_country_idx[j]]

    def sample_products(self, rng, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = int(min(max(k, 1), self.n_prod))
        idx = rng.choice(self.n_prod, size=k, replace=False, p=self.prod_p)
        positions = self.q_off[idx] + (rng.random(k) * self.q_len[idx]).astype(int)
        return idx, self.q_flat[positions].astype(np.int64)


# ---------------------------------------------------------------------------
# Calendar and events
# ---------------------------------------------------------------------------


def expected_orders(model: SyntheticModel, day: date) -> float:
    """Seasonal expectation of the number of orders on ``day`` (weekday x month)."""
    season = model.seasonality
    return max(
        1.0,
        float(season["base_orders_per_day"])
        * float(season["dow_factor"].get(str(day.weekday()), 1.0))
        * float(season["month_factor"].get(str(day.month), 1.0)),
    )


def choose_event(seed: int, target_date: date, probability: float = DEFAULT_EVENT_PROBABILITY) -> str:
    """Deterministically decide whether (and which) event hits ``target_date`` for ``seed``."""
    rng = np.random.default_rng([int(seed), target_date.toordinal(), 1])
    if rng.random() >= probability:
        return "none"
    names = [name for name in EVENTS if name != "none"]
    p = _weights([EVENTS[name]["weight"] for name in names])
    return names[int(rng.choice(len(names), p=p))]


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------


def _timestamp(rng, s: _Sampler, day: date) -> datetime:
    hour = int(rng.choice(np.arange(*BUSINESS_HOURS), p=s.hour_p))
    return datetime(day.year, day.month, day.day, hour, int(rng.integers(0, 60)))


def _make_sale(rng, s: _Sampler, day: date, new_customers: list, day_index: int) -> dict:
    guest = rng.random() < s.guest_prob
    k = s._draw(s.lines_guest if guest else s.lines_known, rng)
    idx, qty = s.sample_products(rng, k)
    if guest:
        customer, country = float("nan"), s.pick_guest_country(rng)
    elif rng.random() < NEW_CUSTOMER_PROB:
        _, country = s.pick_customer(rng)
        customer = float(s.model.max_customer_id + 1 + day_index * NEW_CUSTOMER_BLOCK + len(new_customers))
        new_customers.append(customer)
    else:
        customer, country = s.pick_customer(rng)
    postage = float(rng.choice(POSTAGE_PRICES, p=POSTAGE_P)) if rng.random() < POSTAGE_PROB else None
    return {"kind": "sale", "ts": _timestamp(rng, s, day), "cust": customer, "country": country,
            "idx": idx, "qty": qty, "postage": postage}


def _make_cancel(rng, s: _Sampler, day: date) -> dict:
    idx, qty = s.sample_products(rng, s._draw(s.cancel_lines, rng))
    customer, country = s.pick_customer(rng)
    return {"kind": "cancel", "ts": _timestamp(rng, s, day), "cust": customer, "country": country,
            "idx": idx, "qty": -np.abs(qty), "postage": None}


def _revenue(s: _Sampler, record: dict) -> float:
    return float((record["qty"] * s.prices[record["idx"]]).sum())


def generate_day(model: SyntheticModel, target_date: date, seed: int, event: str,
                 sampler: _Sampler | None = None) -> GeneratedDay:
    """Generate one day of raw (uncleaned) UCI-style rows plus the event ground truth."""
    if event not in EVENTS:
        raise GeneratorError(f"Unknown event '{event}'. Valid: {sorted(EVENTS)}")
    if not is_trading_day(model, target_date):
        raise GeneratorError(f"{target_date} is a non-trading day (weekend/closure per the learned calendar).")

    s = sampler or _Sampler(model)
    scale = model.scale
    day_index = max(1, (target_date - model.hist_end).days)
    ordinal = target_date.toordinal()
    rng = np.random.default_rng([int(seed), ordinal, 0])    # baseline + sampling
    prng = np.random.default_rng([int(seed), ordinal, 2])   # event parameters only

    multiplier = float(np.exp(rng.normal(0, s.sigma) - s.sigma**2 / 2)) if s.sigma > 0 else 1.0
    n_base = max(1, int(round(expected_orders(model, target_date) * multiplier)))
    cancel_rate = float(s.cancel_rates[rng.integers(len(s.cancel_rates))])

    n_orders, min_cancels, target_rev, params = n_base, 0, None, {}
    if event == "demand_spike":
        k = float(prng.uniform(4.5, 6.0))
        target_rev = scale["revenue_median"] + k * max(scale["revenue_sigma"], 1.0)
        n_orders = max(n_base + 1, int(np.ceil(target_rev / scale["revenue_per_order"])))
        params = {"severity_robust_sigma": round(k, 2), "target_revenue": round(target_rev, 2)}
    elif event == "return_spike":
        cancel_rate = float(prng.uniform(0.70, 1.00))
        min_cancels = 3
        params = {"cancel_rate": round(cancel_rate, 3)}
    elif event == "demand_drop":
        k = float(prng.uniform(4.5, 6.0))
        target_orders = max(1, int(round(scale["orders_median"] - k * max(scale["orders_sigma"], 1.0))))
        n_orders = max(1, min(n_base, target_orders))
        params = {"severity_robust_sigma": round(k, 2), "target_orders": target_orders}
    elif event == "promo_uplift":
        factor = float(prng.uniform(1.30, 1.60))
        n_orders = max(n_base + 1, int(round(n_base * factor)))
        params = {"volume_factor": round(factor, 3)}
    elif event == "bulk_order":
        params = {"revenue_multiple_of_baseline": round(float(prng.uniform(1.2, 3.0)), 3)}

    new_customers: list[float] = []
    records = [_make_sale(rng, s, target_date, new_customers, day_index) for _ in range(n_orders)]
    day_rev = sum(_revenue(s, r) for r in records)

    if target_rev is not None:  # demand_spike: top up until the revenue target is met
        for _ in range(MAX_TOPUP_ORDERS):
            if day_rev >= target_rev:
                break
            record = _make_sale(rng, s, target_date, new_customers, day_index)
            records.append(record)
            day_rev += _revenue(s, record)

    if event == "bulk_order":
        pid = int(s.cheap_idx[int(rng.choice(len(s.cheap_idx), p=s.cheap_p))])
        line_rev = max(day_rev * params["revenue_multiple_of_baseline"], day_rev * 1.2)
        qty = max(1, int(np.ceil(line_rev / s.prices[pid])))
        customer, country = s.pick_customer(rng)
        records.append({"kind": "bulk", "ts": _timestamp(rng, s, target_date), "cust": customer,
                        "country": country, "idx": np.array([pid], dtype=int),
                        "qty": np.array([qty], dtype=np.int64), "postage": None})
        params.update({"product": str(s.descs[pid]), "quantity": qty,
                       "line_revenue": round(qty * s.prices[pid], 2)})

    n_sales = len(records)
    if event == "return_spike":
        n_cancels = max(min_cancels, int(round(cancel_rate * n_sales)))
    else:
        n_cancels = max(min_cancels, int(rng.poisson(cancel_rate * n_sales)))
    records.extend(_make_cancel(rng, s, target_date) for _ in range(n_cancels))

    if len(records) >= INVOICE_BLOCK:
        raise GeneratorError(f"{len(records)} invoices exceeds the {INVOICE_BLOCK} per-day identifier block.")

    order = sorted(range(len(records)), key=lambda i: (records[i]["ts"], i))
    base_no = model.max_invoice_no + 1 + day_index * INVOICE_BLOCK

    columns: dict[str, list] = {name: [] for name in REQUIRED_COLUMNS}

    def add(invoice, code, desc, quantity, ts, price, customer, country):
        for name, value in zip(REQUIRED_COLUMNS, (invoice, code, desc, quantity, ts, price, customer, country)):
            columns[name].append(value)

    for seq, record_index in enumerate(order):
        record = records[record_index]
        number = base_no + seq
        invoice = f"C{number}" if record["kind"] == "cancel" else str(number)
        for pid, quantity in zip(record["idx"], record["qty"]):
            add(invoice, s.codes[pid], s.descs[pid], int(quantity), record["ts"], float(s.prices[pid]),
                record["cust"], record["country"])
        if record["postage"] is not None:
            add(invoice, "POST", "POSTAGE", 1, record["ts"], record["postage"], record["cust"], record["country"])

    raw = pd.DataFrame(columns)
    raw["Quantity"] = raw["Quantity"].astype("int64")
    raw["UnitPrice"] = raw["UnitPrice"].astype("float64")
    raw["CustomerID"] = raw["CustomerID"].astype("float64")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])

    spec = EVENTS[event]
    info = {
        "name": event,
        "params": params,
        "expected_detectable": spec["detectable"],
        "expected_signal": spec["signal"],
        "baseline_orders": int(n_base),
        "sale_invoices": int(n_sales),
        "cancel_invoices": int(n_cancels),
        "new_customers": len(new_customers),
    }
    return GeneratedDay(raw=raw, info=info)


# ---------------------------------------------------------------------------
# Partition discovery and cleanup
# ---------------------------------------------------------------------------


def latest_partitioned_date(partitioned_dir: Path) -> date | None:
    """Most recent date that has a product_sales partition, or None."""
    base = Path(partitioned_dir) / paths.PRODUCT_SALES
    found = []
    for file in base.glob("year=*/month=*/day=*/orders.parquet"):
        try:
            year, month, day = (int(part.split("=", 1)[1]) for part in file.parts[-4:-1])
            found.append(date(year, month, day))
        except ValueError:
            continue
    return max(found) if found else None


def _partition_exists(partitioned_dir: Path, day: date) -> bool:
    return (Path(partitioned_dir) / paths.partition_relpath(paths.PRODUCT_SALES, day) / "orders.parquet").exists()


def _remove_day(partitioned_dir: Path, day: date) -> None:
    for dataset in (paths.PRODUCT_SALES, paths.CANCELLATIONS):
        directory = Path(partitioned_dir) / paths.partition_relpath(dataset, day)
        for file in directory.glob("*"):
            if file.is_file():
                file.unlink()
        current = directory
        for _ in range(3):  # day -> month -> year; never removes the dataset directory itself
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def _validate_seed(seed) -> None:
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not (0 <= int(seed) < 2**63):
        raise GeneratorError(f"seed must be a non-negative integer, got {seed!r}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def advance_one_day(
    root,
    seed=None,
    event=None,
    target_date=None,
    overwrite=False,
    event_probability=DEFAULT_EVENT_PROBABILITY,
) -> dict:
    """
    Generate, clean and persist one synthetic trading day under ``<root>/data``.

    With no ``target_date`` the next trading day after the latest partition is
    generated. With no ``seed`` a fresh one is drawn (and returned). With no
    ``event`` one is chosen deterministically from the seed. All validation
    happens before anything is written, and a failure while writing removes any
    partial output.
    """
    root = Path(root)
    _validate_seed(seed)
    if not 0.0 <= float(event_probability) <= 1.0:
        raise GeneratorError("event_probability must be within [0, 1]")
    if event is not None and event not in EVENTS:
        raise GeneratorError(f"Unknown event '{event}'. Valid: {sorted(EVENTS)}")
    if target_date is not None and (not isinstance(target_date, date) or isinstance(target_date, datetime)):
        raise GeneratorError("target_date must be a datetime.date value.")

    model = load_or_build_profile(root)
    partitioned = paths.partitioned_dir(root)
    latest = latest_partitioned_date(partitioned)
    if latest is None:
        raise GeneratorError(f"No partitioned data under {partitioned}; seed the historical data first.")
    if latest < model.hist_end:
        raise GeneratorError(
            f"Historical data is incomplete (latest partition {latest} < historical end {model.hist_end}). "
            "Seed the FULL history before generating."
        )

    skipped: list[str] = []
    if target_date is None:
        target, skipped = next_trading_day(model, latest)
    else:
        target = target_date
    if target <= model.hist_end:
        raise GeneratorError(f"{target} is on or before the historical end ({model.hist_end}); real data is never overwritten.")
    if not is_trading_day(model, target):
        raise GeneratorError(f"{target} is a non-trading day (weekend/closure per the learned calendar).")
    if target > latest + timedelta(days=MAX_DAYS_AHEAD):
        raise GeneratorError(f"{target} is more than {MAX_DAYS_AHEAD} days past the latest data ({latest}).")
    if _partition_exists(partitioned, target) and not overwrite:
        raise GeneratorError(f"A partition for {target} already exists; pass overwrite=True to replace it.")

    used_seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2**32))
    used_event = event if event is not None else choose_event(used_seed, target, event_probability)

    day = generate_day(model, target, used_seed, used_event)
    if day.raw.empty:
        raise GeneratorError("Generator produced zero raw rows.")
    if not (day.raw["InvoiceDate"].dt.date == target).all():
        raise GeneratorError(f"Generated timestamps escaped target day {target}.")

    cleaned = clean(day.raw)
    report = cleaned["report"]
    if not report["reconciliation_ok"]:
        raise GeneratorError(f"Cleaning reconciliation failed for generated data: {report}")
    sales, cancellations = cleaned["product_sales"], cleaned["cancellations"]
    if sales.empty:
        raise GeneratorError("Generated day contains no product sales.")

    manifest = {
        "is_synthetic": True,
        "date": target.isoformat(),
        "seed": used_seed,
        "event": day.info,
        "generator_version": GENERATOR_VERSION,
        "profile_fingerprint": model.profile_fingerprint,
        "historical_end_date": model.hist_end.isoformat(),
        "latest_available_partition": latest.isoformat(),
        "skipped_closed_days": skipped,
        "product_sales_rows": int(len(sales)),
        "cancellation_rows": int(len(cancellations)),
        "revenue": round(float(sales["line_revenue"].sum()), 2),
        "order_count": int(sales["InvoiceNo"].nunique()),
        "return_count": int(cancellations["InvoiceNo"].nunique()),
        "clean_report": report,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    files = [paths.partition_dir(root, paths.PRODUCT_SALES, target) / "orders.parquet"]
    if not cancellations.empty:
        files.append(paths.partition_dir(root, paths.CANCELLATIONS, target) / "orders.parquet")
    mpath = manifest_path(root, target)
    files.append(mpath)

    try:
        if overwrite:
            _remove_day(partitioned, target)
            mpath.unlink(missing_ok=True)
        write_daily_partitions(sales, partitioned, paths.PRODUCT_SALES)
        if not cancellations.empty:
            write_daily_partitions(cancellations, partitioned, paths.CANCELLATIONS)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    except Exception:
        _remove_day(partitioned, target)
        mpath.unlink(missing_ok=True)
        raise

    return {
        "status": "advanced",
        "date": target.isoformat(),
        "rows_ingested": int(len(sales)),
        "source": "synthetic",
        "seed": used_seed,
        "event": used_event,
        "expected_detectable": EVENTS[used_event]["detectable"],
        "skipped_closed_days": skipped,
        "files": [f.relative_to(root).as_posix() for f in files],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one synthetic trading day into <root>/data/partitioned.")
    parser.add_argument("--root", type=Path, default=paths.PROJECT_ROOT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--event", choices=sorted(EVENTS))
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        advance_one_day(args.root, seed=args.seed, event=args.event, target_date=args.date, overwrite=args.overwrite),
        indent=2,
    ))
