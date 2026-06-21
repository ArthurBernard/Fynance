# 4 — Subpackages: map & stability policy

What each subpackage exposes, and the **policy** that governs changes to it — the
single most useful thing to know before touching code here, because the packages
have very different change budgets (some are frozen public API, some are open for
modernisation).

## Policy matrix

| Subpackage | Policy | Why |
|---|---|---|
| `core` | **Change with care** (wide blast radius) — `PriceSeries` + protocols | The seams everything composes through |
| `data` | **Extend freely** — the only I/O layer (ports & adapters) | New sources/formats are additive |
| `features` / `metrics` | **Extend freely** — numerical kernels are Numba `@njit` | Fast, correct, depended on |
| `portfolio.allocation` | **Stable public API** — deprecation path required for breaking changes | Users call ERC/HRP/IVP/MDP/MVP directly |
| `estimator` / `models.econometric_models` | **Single Numba implementation** — do not duplicate parameter logic | One source for ARMA/GARCH estimation |
| `signal` / `models` | **Modernise freely** (PyTorch) | The active R&D surface |
| `backtest` / `plot` / `strategy` | **Improve freely** | Engine/reporting/orchestration, no frozen contract |
| `research` | **Extend freely — but stay data-agnostic & result-free** | The AI-driven R&D harness; must never depend on real data or store results (→ `output_dir` only) |

## Public API surface (what callers import)

- **`core`** — `PriceSeries`; the protocols (`DataSource`/`FeatureTransform`/
  `SignalModel`/`Allocator`/`CostModel`/`Metric`).
- **`data`** — `load()`, `CSVSource`/`ParquetSource`, `align`/`resample`,
  `train_test_split`/`walk_forward`.
- **`features`** — momentums (EMA/SMA/WMA…), indicators (RSI/MACD/Bollinger/…),
  `filters`, `scale`, `engineering`, `regime`, `money_management`. Numba `@njit`
  kernels live in the `.py` modules (no `_cy` twins).
- **`metrics`** — `sharpe`, `sortino`, `calmar`, `diversified_ratio`,
  `annual_return`/`annual_volatility`, `drawdown`/`mdd`, `perf_*`, `roll_*`
  variants, plus one-call `summary`.
- **`signal` / `portfolio`** — `sign`/`threshold`/`rank`/vol-target mappers +
  anti-churn `ema_smooth`/`deadband`/`min_hold` + `SignalPipeline`; allocation
  (`ERC`/`HRP`/`IVP`/`MDP`/`MVP`, `rolling_allocation()`) and sizing
  (`kelly_fraction`/`vol_target`/`transaction_cost`).
- **`models`** — econometric (`ARMA`/`GARCH` family via `get_parameters`) and
  neural (`MultiLayerPerceptron`, `RollMultiLayerPerceptron`, RNN/`GRU`/`LSTM`,
  attention, `TemporalConvNet`, `Transformer`), `StackingEnsemble`; custom losses
  under `models/loss/` (Sharpe/Sortino/Calmar/Omega/directional/hybrid);
  `ObjectiveModel` (objective-aligned training — net-of-cost, mini-batch) in
  `models/objective.py`; training utils in `models/training.py`.
- **`backtest` / `plot` / `strategy`** — `backtest()` + `BacktestResult` +
  `ProportionalCost`; `tearsheet`/`tearsheet_text`; `Strategy` +
  `run_walk_forward`. (The legacy live-viz objects `PlotBackTest`/
  `DynaPlotBackTest`/`display_perf` remain as lazy submodules, off the eager
  surface.)
- **`research`** (namespaced as `fynance.research.*`, not flattened) — `Experiment`,
  `run_experiment`, `write_report`, `gbm`/`regime_switching`. Driven by the
  user-level `/run-strategy` skill. Artifacts go only to a caller-provided
  `output_dir`.

## Known sharp edges (by design)

- **Performance metrics live in `fynance.metrics`** (since 2.0), not
  `fynance.features` — `ratios.py`/`drawdown.py`/`returns.py`/`summary.py`. The
  Numba metric kernels are in `features/_metrics_helpers.py`; `mad`/`roll_mad`
  are in `features/stats.py`.
- **`estimator.estimation()`** is an experimental stub that raises
  `NotImplementedError` — use `models.econometric_models.get_parameters`
  (Numba-backed) for ARMA/GARCH estimation.
- **`lstm.py`/`gru.py` internal hierarchies** (`_LSTMCell → LSTMCell →
  LongShortTermMemory`) and `_RollingBasis`/`RollMultiLayerPerceptron` are
  intentionally **not** split — too tightly coupled.
- **numpy is the lingua franca** at every seam; PyTorch is confined to `models/`;
  table inputs are coerced to numpy at the `data/` edge (no pandas in the core).

> Open work lives in `07-roadmap.md`; this file only notes settled design points
> so an agent doesn't mistake one for a bug.
