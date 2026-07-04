---
plan: v2-refactor/E8-strategy
kind: leaf
status: done
complexity: medium
deps: [1]
parallel: false
---

# E8.02 — walk-forward strategy run

Add `Strategy.run_walk_forward(data, *, train, test, step, purge=0)` using the E2
`walk_forward` splitter: refit features+model per window on train only, predict on
test, stitch out-of-sample positions, backtest the concatenated OOS series.

## Guarantees
- features/model are **fit on train slice only** each window (no-lookahead);
  OOS positions never use data past their timestamp; stitched series is contiguous.

## Files
- extend `fynance/strategy/strategy.py`; `tests/strategy/test_walk_forward.py`.

## Test
- OOS coverage == expected window union; a leakage probe (corrupting future train
  data must not change earlier OOS positions); summary finite.

## Done when
- walk-forward run green with an explicit no-lookahead probe; closes E8.
