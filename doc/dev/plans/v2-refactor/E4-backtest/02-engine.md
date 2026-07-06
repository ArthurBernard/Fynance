---
plan: v2-refactor/E4-backtest
kind: leaf
status: done
complexity: high
deps: [1]
parallel: false
---

# E4.02 — vectorized backtest engine

`fynance/backtest/engine.py` — the core function. Pure numpy, vectorized.

## API
- `backtest(prices_or_returns, positions, *, cost=None, rebalance="step",
  capital=1.0, returns_input=False) -> BacktestResult`
  - infer returns from prices (pct) unless `returns_input`.
  - **causal**: position at t applies to return at t+1 (shift positions by 1);
    assert positions cannot use future returns.
  - gross strat returns = `pos_shifted * asset_returns` (sum over assets if 2-D).
  - net returns = gross − `cost(weights)` per step.
  - equity = `capital * cumprod(1 + net)`.
- single-asset (1-D) and multi-asset (2-D, weights) both supported.

## Files
- new `fynance/backtest/engine.py`
- new `fynance/tests/backtest/test_engine.py`

## Test
- buy-and-hold (pos=1) equity == price path normalized (atol 1e-12);
- zero position → flat equity == capital;
- **no-lookahead**: shuffling returns *after* the last position date doesn't change
  equity up to that date; a position known at t cannot capture r_t;
- costs reduce net returns by exactly the turnover cost;
- 2-D weights case matches manual computation on a tiny example.

## Done when
- engine green incl. causality + cost assertions; numpy-only; mypy clean.
