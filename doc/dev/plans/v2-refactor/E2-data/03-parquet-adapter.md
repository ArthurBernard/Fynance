---
plan: v2-refactor/E2-data
kind: leaf
status: done
complexity: low
deps: [1, 2]
parallel: true
---

# E2.03 — Parquet adapter

`fynance/data/parquet.py` — same contract as CSV via `pl.read_parquet`. Share the
column-selection / datetime-index logic with the CSV adapter (extract a helper in
`data/base.py` if it isn't already).

## Files
- new `fynance/data/parquet.py`; register `"parquet"`
- `fynance/tests/data/test_parquet.py` (+ tiny parquet fixture written in-test)

## Test
- write a small DataFrame to parquet in a tmp_path, load it back → matches the
  CSV adapter's behaviour on the same data.

## Done when
- `load("x.parquet")` returns a `PriceSeries`; tests green.
