---
plan: v2-refactor
kind: global
status: planning
roadmap: "fynance 2.0 — full refactor into a layered ML/DL backtesting tool (data→features→signal→portfolio→backtest→metrics)"
release_on_done: true
---

# fynance 2.0 — global plan

A clean, breaking **2.0** that turns the current *toolbox* into a *complete tool*
for backtesting ML/DL trading strategies — from data ingestion to performance
reporting, through signal prediction and portfolio construction — **while staying
modular** (composable units, never a forced pipeline).

## Vision

```
data → features → signal (ML/DL prediction → position) → portfolio (alloc + sizing)
     → backtest (vectorized) → metrics / reporting
```

Each maillon is an independent, reusable unit with a clear contract. An optional
`Strategy` orchestrator composes them; it is never mandatory.

## Architecture decisions (settled in brainstorm)

1. **Layered + Protocols, ports&adapters only at I/O.** Not full hexagonal. The
   domain is maths on arrays; we use `typing.Protocol` seams for internal
   composition (`FeatureTransform`, `SignalModel`, `Allocator`, `CostModel`,
   `Metric`) and ports&adapters **only** where the outside world is touched
   (`DataSource` now; `Broker`/execution later).
2. **numpy is the lingua franca** at every seam. **pytorch is confined to
   `models/`** (entered at `set_data`, exited at `predict` → numpy); used only
   where autodiff (differentiable Sharpe/Sortino/… losses) or GPU pays.
   **numba `@njit`** for hot CPU loops (estimator, rolling).
3. **Composition over a god-object.** `PriceSeries` is a *thin* numpy-backed
   value object (compose, **never subclass** `np.ndarray`): a handful of core
   price↔return conversions + `.pipe()` delegating to free functions. The rich
   transforms stay free functions in `features/`.
4. **No backward-compat.** Drop legacy shims, drop Cython (→ numba), drop the
   `*_cy` duality. This is a 2.0; breaking is expected.
5. **UI last, optional.** A thin Streamlit playground (`fynance[ui]`, `apps/`)
   built on a clean `tearsheet()` API — not a bespoke IDE. The notebook is the
   reference workflow.

## Target layout

```
fynance/
  core/       PriceSeries value object · Protocols · numpy/torch bridges
  data/       DataSource port · CSV/Parquet adapters · align/resample · temporal splits
  features/   transforms + indicators (perf-metrics removed)
  metrics/    performance metrics (moved out of features)
  models/     pytorch models · numba estimator (ex-Cython)
  signal/     prediction → position mappers
  portfolio/  allocation + sizing (ex-algorithms)
  backtest/   vectorized engine: positions + prices + costs → BacktestResult
  plot/       reporting · tearsheet()
  strategy/   optional orchestrator composing the protocols
apps/         streamlit playground (optional extra)
```

## Epics & dependency order

| Epic | Scope | Depends on |
|------|-------|-----------|
| **E1 core** | `PriceSeries`, Protocols, array bridges | — (foundation) |
| **E2 data** | DataSource port, CSV/Parquet adapters, align, splits | E1 |
| **E3 features+metrics** | regroup features, extract perf-metrics → `metrics/` | E1 |
| **E4 backtest** | CostModel + vectorized engine → `BacktestResult` | E1 |
| **E5 reporting** | `metrics/` consolidation, `plot/`, `tearsheet()` | E3, E4 |
| **E6 signal+portfolio** | `signal/` mappers, `portfolio/` (ex-algorithms) | E1 |
| **E7 models+numba** | estimator → numba, drop Cython, SignalModel conformance | E1 |
| **E8 strategy** | optional orchestrator + walk-forward run | E2,E3,E4,E6,E7 |
| **E9 ui** | Streamlit playground (optional) | E5, E8 |
| **E10 docs** | Sphinx restructure, example notebook, README/migration | all |

Critical path: **E1 → (E2 ∥ E3 ∥ E6 ∥ E7) → E4 → E5 → E8 → (E9 ∥ E10)**.

## Execution protocol

- Each leaf ships as **one small disposable PR** into `develop` (CLAUDE.md: one
  PR = one concern). Order respects `deps:`.
- Every leaf must keep the 4 CI gates green (tests 3.10–3.13, ruff+interrogate,
  Sphinx `-W`, mypy) and respect **no-lookahead** (property tests).
- Because 2.0 breaks the public API, land the breaking moves early and update
  docs per-epic; the final `metrics`/layout doc overhaul is E10.
- Release the whole tree as **v2.0.0** once E1–E8 are in (E9/E10 may trail into
  2.0.x / 2.1).

## Open decision points (flagged in leaves)

- **D1** `features/*_cy` (momentums/metrics Cython): migrate to numba in 2.0, or
  keep frozen? Leaning migrate (consistency, no compile step). → E7.
- **D2** `PriceSeries` vs a multi-column `Panel` for OHLCV: ship `PriceSeries`
  (mono) first; `Panel` deferred (enables ATR/ADX/OBV/VWAP later). → E1/E2.
- **D3** Backtest costs: proportional only in 2.0; non-linear slippage deferred.
- **D4** `Strategy` API style: fluent builder vs declarative config. → E8 leaf 01.
