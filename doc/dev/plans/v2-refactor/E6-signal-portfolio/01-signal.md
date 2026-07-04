---
plan: v2-refactor/E6-signal-portfolio
kind: leaf
status: done
complexity: medium
deps: []
parallel: true
---

# E6.01 — `signal/` package

New `fynance/signal/` — the bridge prediction → position. Pure numpy, causal.

## API (mappers: prediction array → position array in [-1,1] or weights)
- `sign(pred)` — long/short by sign.
- `threshold(pred, *, long=0.0, short=0.0)` — flat band.
- `rank(pred, *, top, bottom)` — cross-sectional long/short by rank.
- `vol_target_position(pred, returns, target_vol, ...)` — reuse
  `portfolio.sizing.vol_target` to scale a directional signal (causal).
- optional `SignalPipeline` composing a `SignalModel` (E1.04) + a mapper, exposing
  `predict_position(X)`.

## Files
- new `fynance/signal/*`; new `fynance/tests/signal/test_signal.py`

## Test
- sign/threshold/rank correct on tiny arrays; vol-target scaling is causal
  (no future returns); positions bounded as documented.

## Done when
- `signal/` green; composes with a dummy `SignalModel`; mypy clean.
