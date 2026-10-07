"""
Tests for the cleaning layer. These run on tiny in-memory frames (and a tiny
generated .xlsx), so they do not need the real dataset.
"""

from datetime import datetime

import pandas as pd
import pytest

from pipeline.clean import (CleaningError, clean, load_raw, parse_excel_invoice_dates)


def raw_frame(rows):
    cols = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    return pd.DataFrame(rows, columns=cols)


T = datetime(2011, 3, 4, 10, 30)
ROWS = [
    ("1001", "85123A", "MUG", 6, T, 2.5, 17850.0, "United Kingdom"),     # clean sale
    ("1002", "22423", "CAKE STAND", 2, T, 12.0, None, "France"),          # clean sale, guest
    ("C1003", "85123A", "MUG", -1, T, 2.5, 17850.0, "United Kingdom"),    # cancellation
    ("1004", "POST", "POSTAGE", 1, T, 18.0, 17850.0, "United Kingdom"),   # non-product
    ("1005", "22423", None, 1, T, 12.0, None, "France"),                  # missing description
    ("1006", "22423", "CAKE STAND", -3, T, 12.0, None, "France"),         # data error (negative qty)
    ("1007", "22423", "CAKE STAND", 3, T, 0.0, None, "France"),           # data error (zero price)
]


class TestExcelDateParsing:
    def test_datetime_cells_are_swapped_in_every_year(self):
        # Excel stores true 2010-12-01 as 2010-01-12 and true 2011-10-12 as 2011-12-10.
        s = pd.Series([datetime(2010, 1, 12, 8, 26), datetime(2011, 12, 10, 17, 19)], dtype=object)
        parsed, n = parse_excel_invoice_dates(s)
        assert list(parsed) == [pd.Timestamp(2010, 12, 1, 8, 26), pd.Timestamp(2011, 10, 12, 17, 19)]
        assert n == 2

    def test_text_cells_are_month_first_and_not_swapped(self):
        s = pd.Series(["12/13/2010 9:02", "1/4/2011 10:00"], dtype=object)
        parsed, n = parse_excel_invoice_dates(s)
        assert list(parsed) == [pd.Timestamp(2010, 12, 13, 9, 2), pd.Timestamp(2011, 1, 4, 10, 0)]
        assert n == 0

    def test_mixed_column_keeps_row_alignment(self):
        s = pd.Series([datetime(2010, 1, 12, 8, 26), "12/13/2010 9:02", datetime(2011, 2, 5, 7, 0)], dtype=object)
        parsed, n = parse_excel_invoice_dates(s)
        assert list(parsed) == [pd.Timestamp(2010, 12, 1, 8, 26), pd.Timestamp(2010, 12, 13, 9, 2),
                                pd.Timestamp(2011, 5, 2, 7, 0)]
        assert n == 2

    def test_unparseable_text_becomes_nat_and_is_counted_by_clean(self):
        df = raw_frame([("1", "A1", "X", 1, "not a date", 1.0, 1.0, "UK")])
        out = clean(df)
        assert out["report"]["invalid_invoice_date_rows"] == 1
        assert out["report"]["dropped_data_error_rows"] == 1   # no date -> data error, still reconciled
        assert out["report"]["reconciliation_ok"]

    def test_xlsx_roundtrip_with_mixed_cell_types(self, tmp_path):
        """Real workbook: datetime cells (swapped) next to text cells (correct)."""
        path = tmp_path / "Online Retail.xlsx"
        frame = raw_frame([
            ("1", "A", "X", 1, datetime(2010, 1, 12, 8, 26), 1.0, 1.0, "UK"),      # true 2010-12-01
            ("2", "A", "X", 1, "12/13/2010 9:02", 1.0, 1.0, "UK"),                 # true 2010-12-13
        ])
        frame.to_excel(path, index=False, engine="openpyxl")
        loaded = load_raw(path)
        assert list(loaded["InvoiceDate"]) == [pd.Timestamp(2010, 12, 1, 8, 26), pd.Timestamp(2010, 12, 13, 9, 2)]
        assert clean(loaded)["report"]["repaired_invoice_date_rows"] == 1


class TestClean:
    def test_every_row_lands_in_exactly_one_bucket(self):
        out = clean(raw_frame(ROWS))
        r = out["report"]
        assert r["reconciliation_ok"] and r["rows_accounted_for"] == r["input_rows"] == len(ROWS)
        assert (len(out["product_sales"]), len(out["cancellations"]), len(out["non_product_adjustments"]),
                r["dropped_missing_description"], r["dropped_data_error_rows"]) == (2, 1, 1, 1, 2)

    def test_dropped_error_rows_are_kept_for_audit(self):
        out = clean(raw_frame(ROWS))
        assert sorted(out["dropped_errors"]["InvoiceNo"]) == ["1006", "1007"]

    def test_all_buckets_share_one_schema(self):
        out = clean(raw_frame(ROWS))
        frames = [out["product_sales"], out["cancellations"], out["non_product_adjustments"]]
        assert all(list(f.columns) == list(frames[0].columns) for f in frames)
        assert all((f.dtypes == frames[0].dtypes).all() for f in frames)
        ps = out["product_sales"]
        assert not ps["is_cancellation"].any() and not ps["is_non_product"].any()
        assert out["cancellations"]["is_cancellation"].all() and out["non_product_adjustments"]["is_non_product"].all()
        assert (ps["line_revenue"] == ps["Quantity"] * ps["UnitPrice"]).all()

    def test_empty_buckets_keep_the_schema(self):
        out = clean(raw_frame(ROWS[:1]))
        assert out["cancellations"].empty
        assert list(out["cancellations"].columns) == list(out["product_sales"].columns)
        assert (out["cancellations"].dtypes == out["product_sales"].dtypes).all()

    def test_input_is_not_modified(self):
        df = raw_frame(ROWS)
        before = df.copy()
        clean(df)
        pd.testing.assert_frame_equal(df, before)

    def test_missing_columns_are_rejected(self):
        with pytest.raises(CleaningError, match="Missing required columns"):
            clean(raw_frame(ROWS).drop(columns=["Country"]))

    def test_non_dataframe_is_rejected(self):
        with pytest.raises(CleaningError):
            clean([1, 2, 3])
