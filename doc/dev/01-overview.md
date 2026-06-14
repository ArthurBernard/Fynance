# 1 — Overview

## What fynance is

**fynance** is a Python + Cython library of machine-learning, econometric, and
statistical tools for **financial time-series analysis**. It bundles four things
that usually live in separate packages:

- **Features** — financial indicators, metrics, filters, momentums, scaling
  (Sharpe/Sortino/Calmar, drawdowns, rolling stats, …), with Cython-accelerated
  hot paths.
- **Algorithms** — portfolio allocation (ERC, HRP, IVP, MDP, MVP) and a generic
  rolling/walk-forward driver.
- **Models** — econometric (ARMA/GARCH via a Cython estimator) and neural
  (MLP, RNN/GRU/LSTM, attention) with a walk-forward training base, plus custom
  financial loss functions (Sharpe/Sortino/directional) in PyTorch.
- **Backtest** — evaluation, P&L/perf plotting (static + dynamic), stat printing.

The throughline is **strict temporal causality**: every rolling feature and every
training window is computed from the past only — no lookahead. This is the
library's core invariant, enforced in tests.

## Current state (snapshot)

- **Version `1.3.4`** (in `pyproject.toml`, static), released on `master` and
  published to PyPI; `Development Status :: 5 - Production/Stable`.
- **Python 3.10–3.13** (CI matrix). Build is `setuptools` + **Cython 3** +
  **NumPy 2**.
- **~250 tests** under `fynance/tests/` (mirrors the package), **plus doctests**
  run on every module via `--doctest-modules` — docstring examples are part of
  the suite and must stay runnable. `ruff` + `mypy` configured; Sphinx docs build
  (furo).
- **Core stack**: NumPy 2, pandas 2, SciPy, Numba (`@njit`), **PyTorch** (the ML
  backend — Keras/TensorFlow is being retired), XGBoost, matplotlib/seaborn.

## Repo map

```
fynance/
  features/      # indicators, metrics, momentums, filters, scale,
                 #   roll_functions, money_management
                 #   (each hot path has a .py + a compiled *_cy.pyx twin)
  algorithms/    # allocation (ERC/HRP/IVP/MDP/MVP), rolling_allocation
  models/        # econometric_models (ARMA/GARCH) + neural (mlp/rnn/gru/lstm/
                 #   attention) on a rolling/walk-forward base; loss/ (torch losses)
  estimator/     # estimator_cy.pyx — ARMA/GARCH parameter estimation (Cython)
  backtest/      # plotting (static + dynamic), loss, print_stats
  core/          # series helpers
  _exceptions.py # ArraySizeError and friends (shared error types)
  tests/         # mirrors the package; pytest + doctests
doc/
  source/        # Sphinx (furo) end-user docs
  dev/           # THIS folder — agent-facing brief
setup.py         # Cython extension build only (metadata is in pyproject.toml)
```

## Three things an agent should never break

1. **No lookahead bias** — any feature at `t` must be `f(data[..t])`; any training
   window trains on the strict past. See `03-decisions.md` and `05-testing.md`.
2. **The Cython fallback** — `setup.py` compiles `.pyx` if Cython is present, else
   falls back to shipped `.c`. New numeric code uses **Numba `@njit`**, not new
   Cython.
3. **Doctests are tests** — every docstring example runs under
   `--doctest-modules`; a broken example fails CI.
