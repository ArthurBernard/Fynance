---
plan: v2-refactor/E4-backtest
kind: leaf
status: done
complexity: medium
deps: [2]
parallel: false
---

# E4.03 — `BacktestResult` object

`fynance/backtest/result.py` — the engine's output value object, the hand-off to
metrics & reporting.

## Design
- dataclass: `equity`, `returns` (net), `gross_returns`, `positions`, `costs`,
  `index` (carried from input `PriceSeries` if given).
- methods: `.to_numpy()`, `.to_price_series()` (equity as `PriceSeries`),
  `.summary()` → dict of metrics computed via `fynance.metrics` (sharpe, sortino,
  calmar, mdd, annual_return/vol, turnover, hit-rate). No plotting here (E5).

## Files
- new `fynance/backtest/result.py`; update `backtest/__init__.py` (drop the old
  plot-centric `__all__`; plotting moves to `plot/` in E5).
- new `fynance/tests/backtest/test_result.py`

## Test
- `.summary()` keys present + numerically match direct `fynance.metrics` calls;
  `.to_price_series()` equity round-trips; dataclass equality.

## Done when
- `BacktestResult` green; `backtest` package exports engine + result + cost;
  legacy plot modules slated for E5 move (leave a note, don't break Sphinx).
