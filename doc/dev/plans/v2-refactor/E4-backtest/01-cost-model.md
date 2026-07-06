---
plan: v2-refactor/E4-backtest
kind: leaf
status: done
complexity: low
deps: []
parallel: false
---

# E4.01 — CostModel + proportional cost

`fynance/backtest/cost.py` — concretize the `CostModel` protocol (E1.04).

## API
- `ProportionalCost(fee=0.0, slippage=0.0)`; `__call__(weights) -> NDArray` returns
  per-step cost = `(fee+slippage) * turnover`, turnover = `|w_t - w_{t-1}|` summed
  over assets (reuse the turnover logic from `algorithms/sizing.transaction_cost`).
- D3: non-linear slippage deferred — leave a documented extension point.

## Files
- new `fynance/backtest/cost.py`; new `fynance/tests/backtest/test_cost.py`

## Test
- zero fee → zero cost; constant weights → zero turnover cost; known turnover →
  known cost; parity with the existing `transaction_cost`.

## Done when
- `ProportionalCost` conforms to `CostModel`; tests green.
