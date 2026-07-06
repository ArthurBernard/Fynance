---
plan: v2-refactor/E2-data
kind: leaf
status: done
complexity: medium
deps: [2]
parallel: true
---

# E2.04 — alignment & resampling

`fynance/data/align.py` — multi-asset alignment and frequency resampling, all
**causal** (no forward leakage on downsample; explicit fill policy).

## API
- `align(series: dict[str, PriceSeries], *, how="outer"|"inner", fill="ffill"|None)`
  → aligned `dict` (common index) — ffill only uses past values.
- `resample(ps: PriceSeries, freq, agg="last"|"mean"|"ohlc")` — downsample via
  polars group_by_dynamic; last/mean for a value series.

## Files
- new `fynance/data/align.py`
- new `fynance/tests/data/test_align.py`

## Test
- outer align fills gaps with past-only ffill (assert no future value used);
  inner align intersects indices; resample last == period-end value.

## Done when
- align/resample green incl. a no-lookahead assertion on ffill.
