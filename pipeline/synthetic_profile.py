"""
synthetic_profile.py - Synthetic Profile v2.0.

Learns a statistical fingerprint of the real, cleaned history and persists it as
``data/profile/profile.json``. ``synthetic_generator.py`` samples new days from
this profile; nothing else about the real data is needed at generation time.

Profile v2.0 layout (all values plain JSON):

    profile_version        "2.0"
    historical_start/end   ISO dates of the real history
    trading_dows           weekdays (Mon=0) that traded historically
    closed_calendar_days   recurring "MM-DD" closures
    max_invoice_no         highest real invoice number
    max_customer_id        highest real customer id
    scale                  orders/revenue level, spread, qty_cap, ...
    seasonality            base_orders_per_day, dow_factor, month_factor, residual_sigma
    hour_weights           24 weights for the order-time-of-day distribution
    basket                 guest_prob and empirical line-count distributions
    cancellations          daily cancellation-rate samples and line-count distribution
    countries              list of country names (indexed by customers/guest_countries)
    guest_countries        distribution of countries for guest checkouts
    products               [{code, description, price, weight, qty_samples}]
    customers              [{id, country_idx, weight}]
    profile_fingerprint    content hash, verified on every load

The profile is deterministic: the same cleaned data always yields the same
profile (and fingerprint). A missing profile is built on demand; an unreadable,
tampered-with or wrong-version profile is an error, never silently replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.data_rules import YEAR_END_CLOSURE_MONTH_DAYS  # noqa: E402
from pipeline import paths  # noqa: E402

PROFILE_VERSION = "2.0"

MAX_QTY_SAMPLES_PER_PRODUCT = 16
BASKET_QUANTILE_CAP = 0.995  # winsorise freak baskets so one huge invoice cannot dominate sampling
QTY_CAP_QUANTILE = 0.999
CLOSED_DAY_SEARCH_LIMIT = 60


class GeneratorError(RuntimeError):
    """Raised when synthetic generation cannot safely proceed."""


class ProfileError(GeneratorError):
    """Raised when the synthetic profile is missing, unreadable or invalid."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ProfileError(f"Invalid date value: {value!r}") from exc


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ProfileError(f"{name} must be finite, got {value!r}")
    return number


def _fingerprint(payload: dict[str, Any]) -> str:
    """Deterministic content hash, computed over everything except the fingerprint itself."""
    body = {k: v for k, v in payload.items() if k != "profile_fingerprint"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _distribution(values: pd.Series) -> dict[str, list]:
    """Empirical distribution as parallel value/count lists (compact, exact, JSON-friendly)."""
    counts = values.value_counts().sort_index()
    return {"values": [int(v) for v in counts.index], "counts": [int(c) for c in counts.to_numpy()]}


# ---------------------------------------------------------------------------
# Loading and validating the cleaned inputs
# ---------------------------------------------------------------------------

_SALES_COLUMNS = {
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country", "line_revenue",
}


def _load_cleaned(path: Path, label: str, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise ProfileError(f"No cleaned {label} data found at: {path}")
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # parquet raises several unrelated error types
        raise ProfileError(f"Unable to read cleaned {label} data: {path}") from exc
    missing = required - set(df.columns)
    if missing:
        raise ProfileError(f"Cleaned {label} data is missing columns: {sorted(missing)}")
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
    df["sales_date"] = df["InvoiceDate"].dt.date
    return df


# ---------------------------------------------------------------------------
# Profile building
# ---------------------------------------------------------------------------


def _daily_orders(sales: pd.DataFrame) -> pd.DataFrame:
    daily = sales.groupby("sales_date").agg(orders=("InvoiceNo", "nunique"), revenue=("line_revenue", "sum"))
    daily.index = pd.to_datetime(daily.index)
    return daily


def _closed_calendar(traded: set[date], start: date, end: date, trading_dows: list[int]) -> list[str]:
    """
    Recurring MM-DD closures: the explicit year-end rule, plus any date in the
    history that fell on a normally-trading weekday but had no sales. Saturdays
    are handled by ``trading_dows`` and deliberately do not create MM-DD closures.
    """
    closed = set(YEAR_END_CLOSURE_MONTH_DAYS)
    day = start
    while day <= end:
        if day.weekday() in trading_dows and day not in traded:
            closed.add(day.strftime("%m-%d"))
        day += timedelta(days=1)
    return sorted(closed)


def _seasonality(daily: pd.DataFrame) -> dict[str, Any]:
    orders = daily["orders"].astype(float)
    overall = orders.mean()
    dow_factor = (orders.groupby(orders.index.dayofweek).mean() / overall).to_dict()
    adjusted = orders / orders.index.dayofweek.map(dow_factor).to_numpy()
    base = float(adjusted.mean())
    month_factor = (adjusted.groupby(adjusted.index.month).mean() / base).to_dict()
    expected = base * orders.index.dayofweek.map(dow_factor).to_numpy() * orders.index.month.map(month_factor).to_numpy()
    residual = np.log(orders.to_numpy() / expected)
    return {
        "base_orders_per_day": base,
        "dow_factor": {str(int(k)): float(v) for k, v in dow_factor.items()},
        "month_factor": {str(int(k)): float(v) for k, v in month_factor.items()},
        "residual_sigma": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
    }


MAD_SCALE = 1.4826  # makes MAD comparable to a normal std; the same constant anomaly_detection.py uses


def _robust_sigma(values: pd.Series) -> float:
    """Median-absolute-deviation based sigma, on the same scale as the anomaly detector's z-scores."""
    return float(MAD_SCALE * (values - values.median()).abs().median())


def _scale(daily: pd.DataFrame, sales: pd.DataFrame) -> dict[str, float]:
    """
    Daily level and *robust* spread of orders and revenue. Injected-event severity
    is expressed in these robust sigmas so it is directly comparable with the
    detector's robust z-score (a plain std would be inflated by the freak days).
    """
    orders, revenue = daily["orders"].astype(float), daily["revenue"].astype(float)
    return {
        "orders_median": float(max(orders.median(), 1.0)),
        "orders_sigma": float(max(_robust_sigma(orders), 1.0)),
        "revenue_median": float(max(revenue.median(), 0.0)),
        "revenue_sigma": float(max(_robust_sigma(revenue), 1.0)),
        "revenue_per_order": float(max((revenue / orders).median(), 0.01)),
        "qty_cap": int(max(sales["Quantity"].quantile(QTY_CAP_QUANTILE), 1)),
    }


def _hour_weights(sales: pd.DataFrame) -> list[float]:
    per_invoice = sales.drop_duplicates("InvoiceNo")
    counts = per_invoice["InvoiceDate"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    return [float(c) for c in counts.to_numpy()]


def _basket(sales: pd.DataFrame) -> dict[str, Any]:
    lines = sales.groupby("InvoiceNo").agg(n=("StockCode", "size"), guest=("CustomerID", lambda s: s.isna().all()))
    cap = max(int(lines["n"].quantile(BASKET_QUANTILE_CAP)), 1)
    capped = lines["n"].clip(upper=cap)
    guest = lines["guest"].astype(bool)
    return {
        "guest_prob": float(guest.mean()),
        "lines_known": _distribution(capped[~guest] if (~guest).any() else capped),
        "lines_guest": _distribution(capped[guest] if guest.any() else capped),
    }


def _cancellations(cancels: pd.DataFrame, daily_orders: pd.Series) -> dict[str, Any]:
    if cancels.empty:
        return {"rates": [0.0], "lines": {"values": [1], "counts": [1]}}
    cancel_counts = cancels.groupby("sales_date")["InvoiceNo"].nunique()
    cancel_counts.index = pd.to_datetime(cancel_counts.index)
    rates = (cancel_counts.reindex(daily_orders.index, fill_value=0) / daily_orders).clip(0.0, 1.0)
    lines = cancels.groupby("InvoiceNo").size()
    cap = max(int(lines.quantile(BASKET_QUANTILE_CAP)), 1)
    return {"rates": [float(r) for r in rates.round(6).to_numpy()], "lines": _distribution(lines.clip(upper=cap))}


def _qty_samples(quantities: pd.Series, cap: int) -> list[int]:
    """
    Compact empirical quantity distribution for one product. Short histories are
    kept exactly; longer ones are summarised by quantile *midpoints*, so a
    product's single largest order is not over-represented (evenly spaced
    quantiles that include the maximum would give it a 1-in-N chance per draw
    instead of 1-in-n and badly inflate revenue).
    """
    values = quantities.clip(lower=1, upper=cap).to_numpy().astype(np.int64)
    if len(values) <= MAX_QTY_SAMPLES_PER_PRODUCT:
        return sorted(int(v) for v in values)
    levels = (np.arange(MAX_QTY_SAMPLES_PER_PRODUCT) + 0.5) / MAX_QTY_SAMPLES_PER_PRODUCT
    return [int(v) for v in np.quantile(values, levels, method="nearest")]


def _products(sales: pd.DataFrame, qty_cap: int) -> list[dict[str, Any]]:
    total_rows = float(len(sales))
    result = []
    for code, group in sales.groupby("StockCode", sort=True):
        descriptions = group["Description"].dropna()
        description = descriptions.mode().iloc[0] if not descriptions.empty else str(code)
        result.append(
            {
                "code": str(code),
                "description": str(description),
                "price": float(max(group["UnitPrice"].median(), 0.01)),
                "weight": float(len(group) / total_rows),
                "qty_samples": _qty_samples(group["Quantity"], qty_cap),
            }
        )
    return result


def _customers(sales: pd.DataFrame, countries: list[str]) -> list[dict[str, Any]]:
    known = sales[sales["CustomerID"].notna()]
    if known.empty:
        return []
    country_index = {name: i for i, name in enumerate(countries)}
    total_rows = float(len(known))
    result = []
    for customer_id, group in known.groupby("CustomerID", sort=True):
        country = group["Country"].mode().iloc[0]
        result.append(
            {
                "id": int(customer_id),
                "country_idx": country_index[str(country)],
                "weight": float(len(group) / total_rows),
            }
        )
    return result


def build_profile(
    cleaned_sales_path: Path | str,
    cleaned_cancellations_path: Path | str | None,
    output_path: Path | str,
) -> dict[str, Any]:
    """Build the v2.0 profile from the cleaned history and write it atomically to ``output_path``."""
    sales = _load_cleaned(Path(cleaned_sales_path), "product sales", _SALES_COLUMNS)
    if sales.empty:
        raise ProfileError("Cleaned product sales contains no usable rows.")
    if cleaned_cancellations_path and Path(cleaned_cancellations_path).exists():
        cancels = _load_cleaned(Path(cleaned_cancellations_path), "cancellations", {"InvoiceNo", "InvoiceDate"})
    else:
        cancels = pd.DataFrame(columns=["InvoiceNo", "sales_date"])

    traded = set(sales["sales_date"])
    hist_start, hist_end = min(traded), max(traded)
    trading_dows = sorted({d.weekday() for d in traded})

    daily = _daily_orders(sales)
    scale = _scale(daily, sales)
    countries = sorted(sales["Country"].dropna().astype(str).unique())
    guest_rows = sales[sales["CustomerID"].isna()]
    guest_country_counts = (guest_rows if not guest_rows.empty else sales)["Country"].astype(str).value_counts().sort_index()

    invoice_numbers = pd.to_numeric(sales["InvoiceNo"], errors="coerce").dropna()
    customer_ids = sales["CustomerID"].dropna()

    profile: dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "historical_start": hist_start.isoformat(),
        "historical_end": hist_end.isoformat(),
        "trading_dows": trading_dows,
        "closed_calendar_days": _closed_calendar(traded, hist_start, hist_end, trading_dows),
        "max_invoice_no": int(invoice_numbers.max()) if not invoice_numbers.empty else 0,
        "max_customer_id": int(customer_ids.max()) if not customer_ids.empty else 0,
        "scale": scale,
        "seasonality": _seasonality(daily),
        "hour_weights": _hour_weights(sales),
        "basket": _basket(sales),
        "cancellations": _cancellations(cancels, daily["orders"].astype(float)),
        "countries": countries,
        "guest_countries": {
            "idx": [countries.index(c) for c in guest_country_counts.index],
            "weight": [float(w) for w in (guest_country_counts / guest_country_counts.sum()).to_numpy()],
        },
        "products": _products(sales, scale["qty_cap"]),
        "customers": _customers(sales, countries),
        "source": {"rows": int(len(sales)), "unique_products": int(sales["StockCode"].nunique())},
    }
    profile["profile_fingerprint"] = _fingerprint(profile)
    _validate(profile)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(output_path)
    return profile


# ---------------------------------------------------------------------------
# Validation and the runtime model
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "profile_version", "historical_start", "historical_end", "trading_dows", "closed_calendar_days",
    "max_invoice_no", "max_customer_id", "scale", "seasonality", "hour_weights", "basket",
    "cancellations", "countries", "guest_countries", "products", "customers", "profile_fingerprint",
}
_REQUIRED_SCALE = {"orders_median", "orders_sigma", "revenue_median", "revenue_sigma", "revenue_per_order", "qty_cap"}


def _validate(profile: Any) -> None:
    if not isinstance(profile, dict):
        raise ProfileError("Profile root must be a JSON object.")
    version = profile.get("profile_version")
    if version != PROFILE_VERSION:
        raise ProfileError(
            f"Profile version mismatch: expected {PROFILE_VERSION}, found {version!r}. "
            "Rebuild it with: python pipeline/synthetic_profile.py"
        )
    missing = _REQUIRED_KEYS - set(profile)
    if missing:
        raise ProfileError(f"Profile is missing required fields: {sorted(missing)}")
    if profile["profile_fingerprint"] != _fingerprint(profile):
        raise ProfileError("Profile fingerprint does not match its content (file was modified or corrupted).")

    if _as_date(profile["historical_start"]) > _as_date(profile["historical_end"]):
        raise ProfileError("Profile historical_start is after historical_end.")
    dows = profile["trading_dows"]
    if not dows or not all(isinstance(d, int) and 0 <= d <= 6 for d in dows):
        raise ProfileError("trading_dows must be a non-empty list of weekday integers 0-6.")
    scale = profile["scale"]
    if not isinstance(scale, dict) or _REQUIRED_SCALE - set(scale):
        raise ProfileError(f"Profile scale must contain {sorted(_REQUIRED_SCALE)}.")
    for key in _REQUIRED_SCALE:
        _finite(scale[key], f"scale.{key}")
    for key in ("orders_median", "orders_sigma", "revenue_per_order", "qty_cap"):
        if scale[key] <= 0:
            raise ProfileError(f"scale.{key} must be greater than zero.")
    if len(profile["hour_weights"]) != 24:
        raise ProfileError("hour_weights must contain 24 values.")
    if not profile["products"]:
        raise ProfileError("Profile products must be a non-empty list.")
    if not profile["customers"]:
        raise ProfileError("Profile customers must be a non-empty list.")
    if not profile["countries"]:
        raise ProfileError("Profile countries must be a non-empty list.")


@dataclass(frozen=True)
class SyntheticModel:
    """Validated, read-only view of a v2.0 profile used by the generator."""

    profile_version: str
    hist_start: date
    hist_end: date
    trading_dows: frozenset[int]
    closed: frozenset[str]
    max_invoice_no: int
    max_customer_id: int
    scale: dict[str, float]
    seasonality: dict[str, Any]
    hour_weights: list[float]
    basket: dict[str, Any]
    cancellations: dict[str, Any]
    countries: list[str]
    guest_countries: dict[str, list]
    products: list[dict[str, Any]]
    customers: list[dict[str, Any]]
    profile_fingerprint: str

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "SyntheticModel":
        _validate(profile)
        return cls(
            profile_version=profile["profile_version"],
            hist_start=_as_date(profile["historical_start"]),
            hist_end=_as_date(profile["historical_end"]),
            trading_dows=frozenset(profile["trading_dows"]),
            closed=frozenset(profile["closed_calendar_days"]),
            max_invoice_no=int(profile["max_invoice_no"]),
            max_customer_id=int(profile["max_customer_id"]),
            scale=dict(profile["scale"]),
            seasonality=dict(profile["seasonality"]),
            hour_weights=list(profile["hour_weights"]),
            basket=dict(profile["basket"]),
            cancellations=dict(profile["cancellations"]),
            countries=list(profile["countries"]),
            guest_countries=dict(profile["guest_countries"]),
            products=list(profile["products"]),
            customers=list(profile["customers"]),
            profile_fingerprint=str(profile["profile_fingerprint"]),
        )


def load_profile(profile_path: Path | str) -> SyntheticModel:
    """Load and fully validate a persisted profile."""
    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise ProfileError(f"No profile found at: {profile_path}")
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"Profile is unreadable (invalid JSON): {profile_path}") from exc
    except OSError as exc:
        raise ProfileError(f"Profile is unreadable: {profile_path}") from exc
    return SyntheticModel.from_profile(profile)


def load_or_build_profile(root: Path | str, rebuild: bool = False) -> SyntheticModel:
    """
    Load ``<root>/data/profile/profile.json``. If it does not exist (or ``rebuild``
    is set) it is built from the cleaned data under ``root``. An existing profile
    that fails validation raises instead of being silently replaced, so a corrupt
    or outdated profile is noticed rather than quietly changing generator output.
    """
    root = Path(root)
    target = paths.profile_path(root)
    if rebuild or not target.exists():
        sales = paths.product_sales_path(root)
        if not sales.exists():
            raise ProfileError(f"No profile at {target} and no cleaned data at {sales} to build one from.")
        build_profile(sales, paths.cancellations_path(root), target)
    return load_profile(target)


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------


def is_trading_day(model: SyntheticModel, day: date) -> bool:
    """A day trades if its weekday traded historically and its MM-DD is not a recurring closure."""
    day = _as_date(day)
    return day.weekday() in model.trading_dows and day.strftime("%m-%d") not in model.closed


def next_trading_day(model: SyntheticModel, after: date) -> tuple[date, list[str]]:
    """First trading day strictly after ``after`` and the closed days skipped on the way (ISO strings)."""
    skipped: list[str] = []
    day = _as_date(after) + timedelta(days=1)
    for _ in range(CLOSED_DAY_SEARCH_LIMIT):
        if is_trading_day(model, day):
            return day, skipped
        skipped.append(day.isoformat())
        day += timedelta(days=1)
    raise GeneratorError(f"No trading day found within {CLOSED_DAY_SEARCH_LIMIT} days after {after}; calendar is broken.")


def profile_summary(model: SyntheticModel) -> dict[str, Any]:
    return {
        "profile_version": model.profile_version,
        "historical_start": model.hist_start.isoformat(),
        "historical_end": model.hist_end.isoformat(),
        "trading_dows": sorted(model.trading_dows),
        "closed_calendar_days": len(model.closed),
        "products": len(model.products),
        "customers": len(model.customers),
        "max_invoice_no": model.max_invoice_no,
        "profile_fingerprint": model.profile_fingerprint,
    }


if __name__ == "__main__":
    summary = profile_summary(load_or_build_profile(paths.PROJECT_ROOT, rebuild=True))
    print(f"Profile written to {paths.profile_path(paths.PROJECT_ROOT)}")
    print(json.dumps(summary, indent=2))
