# 2 — Architecture

fynance is a **layered numerical library**, not a service. Ports & adapters live
only at the I/O edge (`data/`); everything else passes arrays in and gets arrays
or models out. The structure is by *concern*, wired through `typing.Protocol`
seams, with a shared causal-windowing pattern running through it.

```
research            experiment harness: run_experiment → Experiment → write_report
   │  drives           (data-agnostic; artifacts -> caller's output_dir only)
   ▼
strategy            optional orchestrator (compose the maillons end-to-end)
   │  composes
   ▼
backtest · plot     vectorized engine + cost → BacktestResult; tearsheet reporting
   │  consumes
   ▼
signal · portfolio · models   position mappers, allocation/sizing, walk-forward
   │  build on                training, econometrics
   ▼
features · metrics · estimator   indicators, perf metrics, ARMA/GARCH params
   │  operate on                  (hot paths in Numba @njit)
   ▼
core · data         PriceSeries + protocols; adapters, align, temporal splits
```

## Subpackage by subpackage

(See [`04-subpackages.md`](04-subpackages.md) for the public API surface and the
stability policy per package.)

- **`core/`** — `PriceSeries` (numpy-backed value object; compose, don't subclass
  `ndarray`) and the pipeline protocols (`DataSource`/`FeatureTransform`/
  `SignalModel`/`Allocator`/`CostModel`/`Metric`, `runtime_checkable`).
- **`data/`** — the only I/O layer: `load()` dispatcher + CSV/Parquet adapters,
  causal `align`/`resample`, no-lookahead `train_test_split`/`walk_forward`.
- **`features/`** — numerical kernels: `momentums` (EMA/SMA/…), `indicators`,
  `filters`, `roll_functions`, `scale`, `engineering`, `regime`,
  `_metrics_helpers` (the Numba metric kernels), `money_management`.
- **`metrics/`** — performance/evaluation metrics split by concern: `ratios`
  (Sharpe/Sortino/Calmar/…), `drawdown`, `returns`, `summary` (one-call panel).
- **`signal/` · `portfolio/`** — `sign`/`threshold`/`rank`/vol-target +
  anti-churn (`ema_smooth`/`deadband`/`min_hold`) mappers + `SignalPipeline`;
  `allocation.py` (ERC/HRP/IVP/MDP/MVP + `rolling_allocation()`, stable public
  API) and `sizing.py`.
- **`models/`** — econometric (`econometric_models.py`, Numba ARMA/GARCH) and
  neural (`mlp`, `rnn`, `gru`, `lstm`, `attention`, `tcn`, `transformer`) on a
  rolling/walk-forward base; `loss/` (torch losses), `objective.py`
  (objective-aligned training), `training.py`, `ensemble.py`.
- **`backtest/`** — vectorized `engine` + `cost` + `result`; the legacy live-viz
  plot stack (`plot_backtest`/`dynamic_plot_backtest`/`backtest_neural_net`) is
  kept for `RollMultiLayerPerceptron` but off the eager public surface.
- **`plot/` · `strategy/`** — composable matplotlib figures + `tearsheet` (lazy
  import); the optional `Strategy` orchestrator + `run_walk_forward`.
- **`research/`** — data-agnostic experiment harness: `experiment.py`
  (`Experiment`), `runner.py` (`run_experiment`), `report.py` (`write_report`,
  lazy matplotlib/nbformat), `synthetic.py` (`gbm`/`regime_switching`). Writes
  artifacts only to a caller-provided `output_dir`; no real-data dependency.
- **`estimator/`** — Numba ARMA/GARCH parameter estimation. **Authoritative** —
  parameter logic lives here / in `econometric_models`, not duplicated.

## Three cross-cutting patterns

### 1. Numba kernels with golden-value parity (`features/`, `models/`, `estimator/`)

Performance-critical kernels are private Numba `@njit` functions (e.g.
`_ema`/`_sma` in `momentums`, the `_roll_*` kernels in `_metrics_helpers`, the
`_arma`/`_arma_garch` kernels in `econometric_models`) wrapped by thin public
functions. There is **no Cython** — the former `*_cy.pyx` twins were ported to
Numba in 2.1 (E7) and the build is pure-Python. Each kernel is cross-checked
against an independent NumPy reference *and* a golden value captured from the
former Cython (1e-9/1e-10) by the property/parity tests
(`tests/features/test_property.py`, the `*_parity` tests). New numeric code uses
Numba `@njit`, not Cython. See [`03-decisions.md`](03-decisions.md).

### 2. Rolling / walk-forward (the causal core)

`_RollingBasis` (`models/rolling.py`) is the base for all walk-forward evaluation.
It is an **iterator**: `__call__` sets the window (`n` = train length, `s` = test
length, `r` = roll step); each `__next__` trains on `X[t-n:t]` and predicts on
`X[t:t+s]`. `RollMultiLayerPerceptron` subclasses it; `rolling_allocation()`
(`portfolio/`) replicates the same shape as a decorator for portfolio methods.

This pattern is *the* lookahead guard: a window can only ever see its own past.
Any new model or feature must preserve it.

### 3. Estimator → models pipeline

`estimator/estimator.py` and `models/econometric_models.py` hold the Numba ARMA/
GARCH kernels; `models` wraps them via `get_parameters()`. The wrapper layer never
re-implements parameter estimation — the Numba kernels are the single source.

## ML backend

PyTorch is the ML backend, confined to `models/`. TensorFlow/Keras is fully
retired (none in the package). New architectures (TCN, Transformer, custom losses)
target PyTorch, with `nn.Module` models trained through the walk-forward base and
financial loss functions (Sharpe/Sortino/directional) implemented as pure torch
ops in `models/loss/`. Every NN model conforms to the `SignalModel` protocol
(`fit`/`predict`) so it composes with `strategy.Strategy`.
