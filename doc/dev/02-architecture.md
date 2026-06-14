# 2 — Architecture

fynance is a **layered numerical library**, not a service. There's no I/O layer
and no daemon: callers pass arrays/DataFrames in and get arrays/models out. The
structure is by *concern* (features → algorithms/models → backtest), with a
shared causal-windowing pattern running through all of them.

```
backtest        evaluate + plot results
   │  consumes
   ▼
algorithms · models      allocation, walk-forward training, econometrics
   │  build on
   ▼
features · estimator     indicators, metrics, ARMA/GARCH params (hot paths in Cython)
   │  operate on
   ▼
core                     series helpers, array wrappers
```

## Subpackage by subpackage

(See [`04-subpackages.md`](04-subpackages.md) for the public API surface and the
stability policy per package.)

- **`features/`** — the numerical kernels: `metrics` (Sharpe/Sortino/Calmar,
  drawdowns, accuracy, z-score…), `momentums` (EMA/SMA/…), `indicators`,
  `filters`, `roll_functions`, `scale`, `money_management`. The heavy ones ship a
  `.py` *and* a compiled `*_cy.pyx` twin (see below).
- **`algorithms/`** — `allocation.py` (ERC, HRP, IVP, MDP, MVP portfolio methods;
  stable public API), `rolling.py` / `rolling_allocation()` (walk-forward driver
  for allocation), `browsers` (+ `browsers_cy.pyx`).
- **`estimator/`** — `estimator_cy.pyx`: the Cython ARMA/GARCH parameter
  estimator. **Authoritative** — parameter logic lives here, not duplicated in
  Python.
- **`models/`** — econometric (`econometric_models.py` wrapping the estimator) and
  neural (`mlp`, `rnn`, `gru`, `lstm`, `attention`) on a rolling/walk-forward
  base; `loss/` holds custom PyTorch losses.
- **`backtest/`** — `plot_backtest`, `dynamic_plot_backtest`, `print_stats`,
  `loss`; evaluation and visualisation. Improve freely.
- **`core/`** — `series.py` array/series helpers shared across packages.

## Three cross-cutting patterns

### 1. Cython / Python dual implementation (`features/`)

Each performance-critical computation exists twice: `metrics_cy.pyx` (Cython,
compiled to a `.so`) and `metrics.py` (pure Python). `features/__init__.py`
imports both. `setup.py`'s `USE_CYTHON='auto'` guard compiles the `.pyx` if Cython
is available, else falls back to pre-compiled `.c` files — **do not break this
fallback**.

> **Going forward**: new performance-critical code uses **Numba `@njit`** in the
> Python file, *not* new Cython. The Cython twins are kept (extend-only), not
> grown. See [`03-decisions.md`](03-decisions.md).

### 2. Rolling / walk-forward (the causal core)

`_RollingBasis` (`models/rolling.py`) is the base for all walk-forward evaluation.
It is an **iterator**: `__call__` sets the window (`n` = train length, `s` = test
length, `r` = roll step); each `__next__` trains on `X[t-n:t]` and predicts on
`X[t:t+s]`. `RollMultiLayerPerceptron` subclasses it; `rolling_allocation()`
(`algorithms/`) replicates the same shape as a decorator for portfolio methods.

This pattern is *the* lookahead guard: a window can only ever see its own past.
Any new model or feature must preserve it.

### 3. Estimator → models pipeline

`estimator/estimator_cy.pyx` estimates ARMA/GARCH parameters; `models/
econometric_models.py` wraps it via `get_parameters()`. The Python layer never
re-implements parameter estimation — the Cython estimator is the single source.

## ML backend

PyTorch is the ML backend. Legacy Keras/TensorFlow code is being **retired, not
extended**: new architectures (TCN, Transformer, custom losses) target PyTorch,
with `nn.Module` models trained through the walk-forward base and financial loss
functions (Sharpe/Sortino/directional) implemented as pure torch ops in
`models/loss/`.
