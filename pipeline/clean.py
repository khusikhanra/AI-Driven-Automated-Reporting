"""
clean.py - Day 1 cleaning and data-quality pipeline for the UCI Online Retail data.

Two entry points:

1. ``python pipeline/clean.py [--raw PATH]``
   Loads the raw file from ``data/raw`` (CSV or Excel), cleans it and writes the
   cleaned parquet files plus a data-quality report to ``data/cleaned``.
2. ``from pipeline.clean import clean``
   Reusable in-memory cleaning. ``synthetic_generator.py`` pushes every generated
   day through this exact function, so synthetic data obeys the same rules as the
   real history.

Every input row lands in exactly one bucket (dropped description / cancellation /
non-product adjustment / dropped data error / clean product sale) and a
reconciliation check enforces that the buckets sum back to the input.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.data_rules import is_cancellation, is_non_product_code  # noqa: E402
from pipeline import paths  # noqa: E402

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

OUTPUT_COLUMNS = REQUIRED_COLUMNS + ["line_revenue", "is_cancellation", "is_non_product"]

# Stable timestamp resolution so historical and synthetic parquet files always
# share one schema regardless of how the raw file was parsed.
TIMESTAMP_DTYPE = "datetime64[us]"

RAW_FILE_CANDIDATES = ("Online Retail.xlsx", "OnlineRetail.xlsx", "OnlineRetail.csv", "Online Retail.csv")


class CleaningError(Exception):
    """Raised when the cleaning pipeline cannot produce valid output."""


# ---------------------------------------------------------------------------
# Date parsing (Excel quirk)
# ---------------------------------------------------------------------------


def parse_excel_invoice_dates(series: pd.Series) -> tuple[pd.Series, int]:
    """
    Parse the InvoiceDate column of the original ``Online Retail.xlsx``.

    The workbook was built from a day/month-ambiguous text export, so its cells
    are of two kinds:

    * **text** cells (e.g. ``"12/13/2010 9:02"``) are unambiguous month-first
      dates and parse correctly;
    * real **datetime** cells have day and month *swapped* (the true date
      2010-12-01 is stored as 2010-01-12). Only dates whose day is <= 12 can end
      up as datetime cells at all, since anything else is not a valid swapped
      date and was kept as text.

    The swap is therefore applied to exactly the datetime-typed cells, for every
    year in the file. A date-range heuristic cannot do this: swapped 2011 dates
    (e.g. true 2011-10-12 stored as 2011-12-10) look perfectly plausible.
    Verified against the independent CSV release of this dataset: 100% of the
    232,959 datetime cells and all 308,950 text cells reproduce its dates.

    Returns ``(parsed, swapped_cell_count)``.
    """
    is_datetime = series.map(lambda v: isinstance(v, (datetime, pd.Timestamp)))
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[us]")

    text = series[~is_datetime]
    if not text.empty:
        parsed.loc[text.index] = pd.to_datetime(text, format="mixed", errors="coerce").astype("datetime64[us]")

    cells = series[is_datetime]
    if not cells.empty:
        stored = pd.to_datetime(cells)
        swapped = pd.to_datetime(
            {
                "year": stored.dt.year,
                "month": stored.dt.day,
                "day": stored.dt.month,
                "hour": stored.dt.hour,
                "minute": stored.dt.minute,
                "second": stored.dt.second,
            },
            errors="coerce",
        )
        parsed.loc[cells.index] = swapped.astype("datetime64[us]")
    return parsed, int(is_datetime.sum())


# ---------------------------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------------------------


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise CleaningError("Missing required columns: " + ", ".join(missing))


def _normalise_types(data: pd.DataFrame) -> pd.DataFrame:
    for column in ("InvoiceNo", "StockCode", "Description", "Country"):
        data[column] = data[column].astype("string").str.strip()
    for column in ("Quantity", "UnitPrice", "CustomerID"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _finalise(frame: pd.DataFrame, *, cancellation: bool, non_product: bool) -> pd.DataFrame:
    """Apply the shared output schema: dtypes plus the two bucket flags."""
    frame = frame.copy()
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce").astype("int64")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce").astype("float64")
    frame["CustomerID"] = pd.to_numeric(frame["CustomerID"], errors="coerce").astype("float64")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce").astype(TIMESTAMP_DTYPE)
    frame["line_revenue"] = (frame["Quantity"] * frame["UnitPrice"]).astype("float64")
    frame["is_cancellation"] = cancellation
    frame["is_non_product"] = non_product
    # Canonical column order, extras dropped: every parquet file shares one schema.
    return frame[OUTPUT_COLUMNS].reset_index(drop=True)


def clean(df: pd.DataFrame) -> dict:
    """
    Clean an in-memory raw UCI-style dataframe.

    Returns a dict with ``product_sales``, ``cancellations``,
    ``non_product_adjustments`` dataframes (all sharing one schema, including
    ``line_revenue``, ``is_cancellation`` and ``is_non_product``), the raw
    ``dropped_errors`` rows (kept so every exclusion can be audited) and a
    data-quality ``report``.

    ``df.attrs["repaired_invoice_date_rows"]`` (set by ``load_raw``) is carried
    into the report; clean() itself never guesses at dates.
    """
    if not isinstance(df, pd.DataFrame):
        raise CleaningError(f"clean() expected pandas DataFrame, got {type(df).__name__}")
    _validate_columns(df)

    data = _normalise_types(df.copy())
    input_rows = len(data)

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], format="mixed", errors="coerce")
    invalid_invoice_date_rows = int(data["InvoiceDate"].isna().sum())
    repaired_invoice_date_rows = int(df.attrs.get("repaired_invoice_date_rows", 0))

    # Rows without a description cannot be labelled or reported on.
    missing_description = data["Description"].isna() | (data["Description"].str.len() == 0)
    dropped_missing_description = int(missing_description.sum())
    data = data.loc[~missing_description]

    # Structurally unusable rows (missing key fields or unparseable numbers) are data
    # errors in every bucket, so each output column has one stable, null-free dtype.
    structural_error = (
        data["InvoiceNo"].isna()
        | (data["InvoiceNo"].str.len() == 0)
        | data["StockCode"].isna()
        | (data["StockCode"].str.len() == 0)
        | data["InvoiceDate"].isna()
        | data["Quantity"].isna()
        | data["UnitPrice"].isna()
    )
    dropped_structural = data.loc[structural_error]
    data = data.loc[~structural_error]

    # Cancellations (documented source-system convention: InvoiceNo starts with 'C').
    cancellation_mask = data["InvoiceNo"].map(is_cancellation).astype(bool)
    cancellations = data.loc[cancellation_mask]
    product = data.loc[~cancellation_mask]

    # Non-product line items (postage, fees, adjustments): explicit documented list.
    non_product_mask = product["StockCode"].map(is_non_product_code).astype(bool)
    non_product_adjustments = product.loc[non_product_mask]
    product = product.loc[~non_product_mask]

    # A genuine product sale needs a positive quantity and price.
    invalid_sale = (product["Quantity"] <= 0) | (product["UnitPrice"] <= 0)
    dropped_errors = pd.concat([dropped_structural, product.loc[invalid_sale]])
    dropped_data_error_rows = len(dropped_errors)
    product_sales = _finalise(product.loc[~invalid_sale], cancellation=False, non_product=False)
    cancellations = _finalise(cancellations, cancellation=True, non_product=False)
    non_product_adjustments = _finalise(non_product_adjustments, cancellation=False, non_product=True)

    missing_customer_rows = int(product_sales["CustomerID"].isna().sum())
    pct_missing_customer_id = (
        round(missing_customer_rows / len(product_sales) * 100, 2) if len(product_sales) else 0.0
    )

    historical_start = historical_end = None
    if not product_sales.empty:
        historical_start = product_sales["InvoiceDate"].min().isoformat(sep=" ")
        historical_end = product_sales["InvoiceDate"].max().isoformat(sep=" ")

    rows_accounted_for = (
        dropped_missing_description
        + len(cancellations)
        + len(non_product_adjustments)
        + dropped_data_error_rows
        + len(product_sales)
    )
    reconciliation_ok = rows_accounted_for == input_rows
    if not reconciliation_ok:
        raise CleaningError(
            f"Cleaning reconciliation failed: input_rows={input_rows}, rows_accounted_for={rows_accounted_for}"
        )

    report = {
        "input_rows": int(input_rows),
        "invalid_invoice_date_rows": invalid_invoice_date_rows,
        "repaired_invoice_date_rows": repaired_invoice_date_rows,
        "dropped_missing_description": dropped_missing_description,
        "cancellation_rows": int(len(cancellations)),
        "non_product_adjustment_rows": int(len(non_product_adjustments)),
        "dropped_data_error_rows": dropped_data_error_rows,
        "clean_product_sales_rows": int(len(product_sales)),
        "missing_customer_id_in_product_sales": missing_customer_rows,
        "pct_missing_customer_id": float(pct_missing_customer_id),
        "rows_accounted_for": int(rows_accounted_for),
        "reconciliation_ok": bool(reconciliation_ok),
        "historical_start": historical_start,
        "historical_end": historical_end,
    }
    return {
        "product_sales": product_sales,
        "cancellations": cancellations,
        "non_product_adjustments": non_product_adjustments,
        "dropped_errors": dropped_errors[REQUIRED_COLUMNS].reset_index(drop=True),
        "report": report,
    }


# ---------------------------------------------------------------------------
# Raw file loading and the historical pipeline
# ---------------------------------------------------------------------------


def find_raw_file(root: Path) -> Path:
    """Locate the raw dataset in ``<root>/data/raw`` (CSV or Excel)."""
    raw_dir = paths.raw_dir(root)
    for name in RAW_FILE_CANDIDATES:
        if (raw_dir / name).exists():
            return raw_dir / name
    others = sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in {".csv", ".xlsx"}) if raw_dir.exists() else []
    if others:
        return others[0]
    raise CleaningError(f"No raw dataset (.csv or .xlsx) found in {raw_dir}")


def load_raw(path: Path) -> pd.DataFrame:
    """
    Read the raw dataset (``.xlsx`` or ``.csv``) with InvoiceDate already parsed.

    Excel files go through :func:`parse_excel_invoice_dates` (cell types are only
    visible here, which is why the date repair lives in the loader). CSV files
    are read as UTF-8, falling back to Latin-1 (the UCI/Kaggle default). The
    number of repaired cells is recorded in ``df.attrs``.
    """
    path = Path(path)
    if not path.exists():
        raise CleaningError(f"Raw file not found: {path}")

    if path.suffix.lower() == ".xlsx":
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise CleaningError("Reading .xlsx needs openpyxl: pip install openpyxl") from exc
        _validate_columns(df)
        df["InvoiceDate"], repaired = parse_excel_invoice_dates(df["InvoiceDate"])
    else:
        dtypes = {"InvoiceNo": "string", "StockCode": "string"}
        try:
            df = pd.read_csv(path, dtype=dtypes, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=dtypes, encoding="latin-1")
        _validate_columns(df)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", errors="coerce")
        repaired = 0
    df.attrs["repaired_invoice_date_rows"] = repaired
    return df


def clean_historical_file(root: Path = paths.PROJECT_ROOT, raw_path: Path | None = None) -> dict:
    """Load the raw dataset, clean it, and write the cleaned outputs under ``root/data/cleaned``."""
    raw_path = Path(raw_path) if raw_path else find_raw_file(root)
    print(f"Loading raw data from: {raw_path}")
    df = load_raw(raw_path)
    print(f"Loaded rows: {len(df):,}")

    result = clean(df)
    report = result["report"]

    out_dir = paths.cleaned_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    result["product_sales"].to_parquet(paths.product_sales_path(root), index=False)
    result["cancellations"].to_parquet(paths.cancellations_path(root), index=False)
    result["non_product_adjustments"].to_parquet(out_dir / "non_product_adjustments.parquet", index=False)
    result["dropped_errors"].to_parquet(out_dir / "dropped_data_errors.parquet", index=False)
    (out_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nData Quality Report\n" + "=" * 50)
    print(json.dumps(report, indent=2))
    print(f"\nCleaned outputs written to {out_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the raw Online Retail dataset.")
    parser.add_argument("--raw", type=Path, help="Raw .csv/.xlsx file (default: auto-detect in data/raw)")
    parser.add_argument("--root", type=Path, default=paths.PROJECT_ROOT)
    args = parser.parse_args()
    clean_historical_file(args.root, args.raw)
