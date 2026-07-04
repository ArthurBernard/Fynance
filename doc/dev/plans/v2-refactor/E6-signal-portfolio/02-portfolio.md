---
plan: v2-refactor/E6-signal-portfolio
kind: leaf
status: done
complexity: medium
deps: []
parallel: true
---

# E6.02 — `portfolio/` package (rename `algorithms/`) + Allocator

Rename `fynance/algorithms/` → `fynance/portfolio/` (2.0 breaking) and express
allocators under the `Allocator` protocol; keep sizing.

## Scope
- `git mv` allocation.py, sizing.py → `portfolio/`; update package `__init__`,
  `__all__`, all imports, tests.
- `Allocator` conformance: `ERC`, `HRP`, `IVP`, `MDP`, `rolling_allocation` expose
  `__call__(cov_or_returns) -> weights`. Keep numpy-only (already pandas-free).
- sizing (`kelly_fraction`, `vol_target`, `transaction_cost`) stays; the cost one
  is also reused by `backtest.cost` (single source — import, don't duplicate).

## Files
- `fynance/portfolio/*` (moved), updated imports across repo + tests under
  `fynance/tests/portfolio/`.

## Test
- allocators isinstance `Allocator`; numeric parity with pre-rename golden values;
  `from fynance.portfolio import ERC, HRP, kelly_fraction` works; old
  `fynance.algorithms` import path is gone (assert ImportError).

## Done when
- portfolio package green; allocation public-API parity (values) preserved.
