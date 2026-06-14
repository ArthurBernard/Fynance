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
- **`algorithms`** — portfolio allocation: `ERC`, `HRP`, `IVP`, `MDP`, `MVP`, and
  `rolling_allocation()` for walk-forward application.
- **`models`** — econometric (`ARMA`/`GARCH` family via `get_parameters`) and
  neural (`MultiLayerPerceptron`, `RollMultiLayerPerceptron`, RNN/`GRU`/`LSTM`,
  attention); custom losses under `models/loss/`.
- **`backtest`** — `BackTest`/plotting objects (`PlotBackTest`,
  `DynaPlotBackTest`, …), `print_stats`.

## Known sharp edges (by design or pending)

- **`models/basis.py`** — `SignalModel`/`MagnitudeModel` are stubs (`pass` /
  `y_pred` never assigned), unreferenced. Slated for removal (see roadmap §6.1).
- **`backtest/dynamic_plot_backtest.py`** — carries a deprecated
  `__BacktestNeuralNet` ("OLD VERSION") and an unused `_BacktestNeuralNet` stub.
- **Large modules pending a split** — `features/metrics.py` (~1 800 lines),
  `models/{attention,econometric_models,rolling}.py`,
  `backtest/dynamic_plot_backtest.py` (~708 lines, 6 classes). See roadmap §5.
- **`lstm.py`/`gru.py` internal hierarchies** (`_LSTMCell → LSTMCell →
  LongShortTermMemory`) are intentionally **not** split — too tightly coupled.

> The "pending" items above live in the (local) `07-roadmap.md`; this file only
> notes them so an agent doesn't mistake a known stub for a bug.
