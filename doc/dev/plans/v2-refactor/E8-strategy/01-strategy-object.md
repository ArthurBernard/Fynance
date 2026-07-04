---
plan: v2-refactor/E8-strategy
kind: leaf
status: done
complexity: high
deps: []
parallel: false
---

# E8.01 — `Strategy` object

`fynance/strategy/strategy.py` — composes a `DataSource`/`PriceSeries` →
`FeatureTransform`(s) → `SignalModel` → signal mapper → `Allocator`/sizing →
`backtest` → `metrics`. Each slot accepts any object conforming to its Protocol;
each slot is **optional** (skip features, skip allocation, etc.).

## Decision D4 — API style
- Decide fluent-builder vs declarative-config. **Recommendation: fluent + dataclass
  config** — `Strategy(features=[...], model=..., signal=..., allocator=..., cost=...)`
  with a `.run(data) -> BacktestResult`. Document the choice in `03-decisions.md`.

## API
- `Strategy(...)` with protocol-typed slots; `.run(data) -> BacktestResult`;
  validates wiring (shapes/contracts) with clear errors; no hidden lookahead
  (features fit on train portion only when used inside walk-forward, E8.02).

## Files
- new `fynance/strategy/*`; `fynance/tests/strategy/test_strategy.py`.

## Test
- an end-to-end tiny run (random/sine data) produces a `BacktestResult` whose
  `.summary()` is finite; swapping any slot for another conforming impl works;
  a missing required slot raises a clear error.

## Done when
- `Strategy(...).run()` green end-to-end; D4 recorded as an ADR.
