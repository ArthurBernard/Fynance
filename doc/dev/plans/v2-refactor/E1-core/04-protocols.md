---
plan: v2-refactor/E1-core
kind: leaf
status: done
complexity: medium
deps: []
parallel: true
---

# E1.04 — Protocol seams

Define the composition contracts in `fynance/core/protocols.py` using
`typing.Protocol` (+ `@runtime_checkable`). These are **structural** — no forced
inheritance; existing/new objects conform by shape.

## Protocols (signatures, all numpy-typed)
- `DataSource` — `load(...) -> PriceSeries` (the only **port**; adapters in E2).
- `FeatureTransform` — `fit(X) -> Self`, `transform(X) -> NDArray`; causal.
- `SignalModel` — `fit(X, y) -> Self`, `predict(X) -> NDArray`.
- `Allocator` — `__call__(returns_or_cov) -> NDArray` (weights).
- `CostModel` — `__call__(weights) -> NDArray` (per-step cost); impl in E4.
- `Metric` — `__call__(returns) -> float`.

## Files
- new `fynance/core/protocols.py`
- export from `fynance/core/__init__.py`
- new `fynance/tests/core/test_protocols.py`

## Test
- `runtime_checkable` isinstance smoke tests with tiny conforming/​non-conforming
  dummies; mypy treats a conforming object as the Protocol (a typing assertion).

## Done when
- protocols importable; runtime_checkable smoke tests green; documented in
  docstrings (each protocol states shapes + causality contract).
