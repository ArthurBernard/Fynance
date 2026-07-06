---
plan: v2-refactor/E3-features-metrics
kind: leaf
status: done
complexity: medium
deps: [1]
parallel: false
---

# E3.02 — regroup features + `FeatureTransform` adoption

Reorganize `features/` into coherent groups and offer a thin class layer
conforming to `FeatureTransform` (E1.04) **wrapping** the existing free functions
— functions remain the primary, composable API.

## Grouping (rename/merge, no behaviour change)
- `transforms.py` — returns/prices/diff/scale/roll z-score (the transform sense).
- `indicators.py` — EMA/MACD/RSI/Bollinger/CCI/HMA/roc/realized_vol/skew/kurt/autocorr.
- `stats.py` — descriptive stats helpers (non-perf).
- `engineering.py`, `regime.py`, `filters.py` — keep.

## FeatureTransform layer
- a small `make_transform(fn, **fixed)` factory + a few first-class classes
  (`ZScore`, `RSI`, …) exposing `fit(X)->self`, `transform(X)->ndarray`, all
  **stateless or fit-on-train-only** (causal). `fit` records only past-derivable
  params (e.g. mean/std on the train slice) — never peeks at transform input future.

## Files
- reorganized `fynance/features/*`; new `features/_transform_base.py`
- updated tests under `fynance/tests/features/`

## Test
- each `FeatureTransform.transform` == its free-function counterpart (parity);
  `isinstance(t, FeatureTransform)` (runtime_checkable); fit-on-train then
  transform-on-test uses **no** test-set statistics (no-lookahead assertion).

## Done when
- regrouped imports stable; transform classes parity-green; mypy clean.
