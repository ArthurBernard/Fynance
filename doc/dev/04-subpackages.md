# 4 — Subpackages: map & stability policy

What each subpackage exposes, and the **policy** that governs changes to it — the
single most useful thing to know before touching code here, because the packages
have very different change budgets (some are frozen public API, some are open for
modernisation).

## Policy matrix

| Subpackage | Policy | Why |
|---|---|---|
| `features` (`.py` + `*_cy.pyx`) | **Extend only** — never rewrite the Cython kernels | Fast, correct, depended on; new code goes in the `.py` twin via Numba |
| `algorithms.allocation` | **Stable public API** — deprecation path required for breaking changes | Users call ERC/HRP/IVP/MDP/MVP directly |
| `estimator` | **Cython is authoritative** — do not duplicate parameter logic in Python | One source for ARMA/GARCH estimation |
| `models` | **Modernise freely** (PyTorch) | The active R&D surface |
| `backtest` | **Improve freely** | Plotting/eval, no external API contract |
| `core` | Shared helpers — change with care (wide blast radius) | Used everywhere |

## Public API surface (what callers import)

- **`features`** — `sharpe`, `sortino`, `calmar`, `drawdown`/`mdd`,
  `roll_*` rolling variants, `z_score`, `accuracy`, momentums (EMA/SMA…),
  filters, `scale`, `roll_functions`, `money_management`. Re-exported from
  `features/__init__.py` (which pulls both the Python and `_cy` implementations).
- **`algorithms`** — portfolio allocation (`ERC`, `HRP`, `IVP`, `MDP`, `MVP`,
  `rolling_allocation()`) and position sizing (`sizing.py`: `kelly_fraction`,
  `vol_target`, `transaction_cost`).
- **`models`** — econometric (`ARMA`/`GARCH` family via `get_parameters`) and
  neural (`MultiLayerPerceptron`, `RollMultiLayerPerceptron`, RNN/`GRU`/`LSTM`,
  attention, `TemporalConvNet`, `Transformer`), `StackingEnsemble`; custom losses
  under `models/loss/` (Sharpe/Sortino/Calmar/Omega/directional/hybrid); training
  utils in `models/training.py`.
- **`backtest`** — `BackTest`/plotting objects (`PlotBackTest`,
  `DynaPlotBackTest`, …), `print_stats`.

## Known sharp edges (by design)

- **`features/metrics.py`** is a thin **re-export aggregator** — the
  implementations live in `returns.py`, `ratios.py`, `drawdown.py`, `stats.py`
  and `_metrics_helpers.py`. Import from `fynance.features` (or
  `fynance.features.metrics`) as before; the split is transparent.
- **`estimator.estimation()`** is an experimental stub that raises
  `NotImplementedError` — use `models.econometric_models.get_parameters`
  (Cython-backed) for ARMA/GARCH estimation.
- **`lstm.py`/`gru.py` internal hierarchies** (`_LSTMCell → LSTMCell →
  LongShortTermMemory`) and `_RollingBasis`/`RollMultiLayerPerceptron` are
  intentionally **not** split — too tightly coupled.
- **Inputs** accept numpy / torch / **polars**; **outputs** are numpy (no pandas).

> Open work lives in `07-roadmap.md`; this file only notes settled design points
> so an agent doesn't mistake one for a bug.
