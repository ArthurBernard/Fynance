---
plan: v2-refactor/E1-core
kind: leaf
status: done
complexity: medium
deps: []
parallel: false
---

# E1.01 — `PriceSeries` value object

Create `fynance/core/price_series.py` defining `PriceSeries`: a **thin, numpy-backed**
container holding a 1-D financial series with its temporal index and metadata.

## Design
- Fields: `values: np.ndarray` (1-D, float64 default), `index: np.ndarray`
  (datetime64 or int positions), `name: str | None`, `freq: str | None`.
- **Composition, not subclassing.** Internally store a contiguous numpy array.
- Construction: `PriceSeries(values, index=None, name=None, freq=None)`; if
  `index` is None, default to a 0..n-1 RangeIndex (int64). Validate lengths match.
- Immutable feel: store a copy / set `values.flags.writeable = False`; mutating
  ops return a **new** `PriceSeries`.
- Dunders: `__len__`, `__getitem__` (int → scalar; slice → new `PriceSeries`
  carrying the sliced index), `__repr__` (head/tail + dtype + freq), `__eq__`
  by value+index, `__array__` (so `np.asarray(ps)` works → numpy interop).
- Constructors: `from_numpy`, `from_polars` (a `pl.Series`/`pl.DataFrame` column,
  picking up a datetime index column if present).

## Files
- new `fynance/core/price_series.py`
- `fynance/core/__init__.py` — export `PriceSeries`
- new `fynance/tests/core/test_price_series.py`

## Test
- round-trip `from_numpy`/`np.asarray` identity; slicing keeps index aligned;
  immutability (writeable False; ops return new); `from_polars` picks datetime
  index; `__len__`/`__getitem__`/`__repr__` doctests.

## Done when
- `PriceSeries` importable from `fynance.core`; tests + doctests green; mypy clean.
