---
plan: v2-refactor/E2-data
kind: leaf
status: done
complexity: medium
deps: [1]
parallel: false
---

# E2.02 — CSV adapter

`fynance/data/csv.py` — read a CSV via polars into `PriceSeries` (mono-column) or
a dict/Panel-lite of `PriceSeries` (multi-column).

## API
- `CSVSource.load(path, *, value_col=None, index_col=None, freq=None) -> PriceSeries`
  — picks the value column (single non-index numeric col if `value_col` None),
  parses `index_col` as datetime (or infers a datetime col), passes `freq`.
- Multi-column: `value_col=None` + several numerics → return `dict[str, PriceSeries]`.
- Robust to: missing header, separator inference (polars), tz-naive datetimes.

## Files
- new `fynance/data/csv.py`; register as `"csv"`
- new fixtures in `fynance/tests/data/` (tiny csv) + `test_csv.py`

## Test
- load a 5-row csv → correct values, datetime index, name; multi-col → dict;
  numeric dtype is float64; round-trip values match the raw file.

## Done when
- `load("x.csv")` returns a `PriceSeries`; tests + doctest green.
