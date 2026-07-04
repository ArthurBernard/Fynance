---
plan: v2-refactor/E3-features-metrics
kind: leaf
status: done
complexity: medium
deps: []
parallel: false
---

# E3.01 — extract performance metrics into `fynance/metrics/`

Move evaluation metrics out of `features/` into a dedicated `fynance/metrics/`.
This is a **breaking move** (2.0): no re-export shim in `features`.

## What moves
From `features/{ratios,drawdown,returns,stats,metrics}.py` →
`metrics/`: `sharpe`, `sortino`, `calmar`, `roll_sharpe`, `mdd`/`drawdown`,
`annual_return`, `annual_volatility`, `perf_strat`, `returns_strat`,
`percent_positive`, `tail_ratio`, etc. (full list = current `features.metrics.__all__`).

What **stays** in features: indicators, momentums, scale/normalization,
engineering, regime, filters, roll_functions, raw `returns`/`prices` transforms
used as *features* (keep a thin `features.returns` for the transform sense if
needed, distinct from the perf metric).

## Layout
- `fynance/metrics/__init__.py` (+ submodules `ratios.py`, `drawdown.py`,
  `summary.py`); each function conforms to the `Metric` protocol where it takes a
  return series → float.
- delete the moved code from `features/`; update `features/__init__.py` `__all__`.
- update all internal imports (backtest, models, tests).

## Files
- new `fynance/metrics/*`; edited `fynance/features/*`; moved tests →
  `fynance/tests/metrics/`.

## Test
- numeric parity with pre-move values on a fixed series (golden values);
  `from fynance.metrics import sharpe, mdd, ...` works; `features` no longer
  exposes them (assert ImportError/AttributeError).

## Done when
- metrics live in `fynance/metrics`; full suite green; doc autosummary updated
  (deferred detail to E10 but keep `-W` Sphinx green now).
