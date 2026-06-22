# 1 — Overview

## What fynance is

**fynance** is a pure-Python library (Numba-accelerated kernels) of
machine-learning, econometric, and statistical tools for **financial time-series
analysis and backtesting**. Since 2.0 it is a **layered backtesting tool**
(data → features → signal → portfolio → backtest → metrics) composed through
`typing.Protocol` seams, not just a grab-bag of functions:

- **Core** — `PriceSeries` (thin numpy-backed value object) and the pipeline
  protocols (`DataSource`/`FeatureTransform`/`SignalModel`/`Allocator`/
  `CostModel`/`Metric`).
- **Data** — `load()` + CSV/Parquet adapters, causal align/resample, no-lookahead
  temporal splits.
- **Features** — financial indicators, momentums, filters, scaling, feature
  engineering, regime detection (Numba-accelerated hot paths).
- **Metrics** — Sharpe/Sortino/Calmar, drawdowns, rolling stats, one-call summary.
- **Signal / Portfolio** — prediction→position mappers; allocation (ERC/HRP/IVP/
  MDP/MVP) + sizing.
- **Models** — econometric (ARMA/GARCH) and neural (MLP, RNN/GRU/LSTM, attention,
  TCN, Transformer, stacking ensemble) on a walk-forward training base, plus custom
  PyTorch losses (Sharpe/Sortino/Calmar/Omega/directional/hybrid).
- **Backtest / Plot / Strategy** — vectorized engine + cost models →
  `BacktestResult`; `tearsheet` reporting; optional `Strategy` orchestrator.
- **Research** — data-agnostic experiment harness (`Experiment`,
  `run_experiment`, `write_report`, synthetic generators) emitting portable result
  artifacts to a caller-provided `output_dir`; fynance never stores results itself.

The throughline is **strict temporal causality**: every rolling feature and every
training window is computed from the past only — no lookahead. This is the
library's core invariant, enforced in tests.

## Current state (snapshot)

- **Version `2.9.0`** (in `pyproject.toml`, static), released on `master` and
  published to PyPI; `Development Status :: 5 - Production/Stable`.
- **Python 3.10–3.13** (CI matrix). Build is **pure-Python** (setuptools, no
  compile step) — numerical kernels are Numba `@njit`. No Cython.
- **~570 unit/benchmark tests** under `fynance/tests/` (mirrors the package),
  **plus doctests** run on every module via `--doctest-modules` — docstring
  examples are part of the suite and must stay runnable (~660 collected in
  total).
- **Four CI gates**: pytest (3.10–3.13), `ruff`, `interrogate` (docstring
  coverage ≥ 80%), Sphinx build `-W`, and `mypy` clean.
- **Core stack**: NumPy, SciPy, Numba (`@njit`), **PyTorch** (the ML backend —
  no TensorFlow/Keras), matplotlib/seaborn (lazy — `import fynance` stays
  matplotlib-free).

## Repo map

```
fynance/
  core/        # PriceSeries value object + pipeline protocols
  data/        # load() + CSV/Parquet adapters, align/resample, temporal splits
  features/    # indicators, momentums, filters, scale, engineering, regime,
               #   _metrics_helpers (Numba kernels), money_management
  metrics/     # ratios, drawdown, returns, summary (perf/eval metrics)
  signal/      # prediction→position mappers + SignalPipeline
  portfolio/   # allocation (ERC/HRP/IVP/MDP/MVP) + sizing
  models/      # econometric_models (ARMA/GARCH, Numba) + neural (mlp/rnn/gru/
               #   lstm/attention/tcn/transformer) on a walk-forward base;
               #   loss/ (torch losses), training, ensemble
  backtest/    # vectorized engine + cost + result; legacy live-viz plot stack
  plot/        # composable matplotlib figures + tearsheet (lazy import)
  strategy/    # optional Strategy orchestrator + walk-forward run
  research/    # data-agnostic experiment harness: Experiment, run_experiment,
               #   write_report, synthetic generators (results -> output_dir only)
  estimator/   # ARMA/GARCH parameter estimation (Numba)
  tests/       # mirrors the package; pytest + doctests
doc/
  source/      # Sphinx (furo) end-user docs
  dev/         # THIS folder — agent-facing brief
pyproject.toml # authoritative build config (pure-Python; no setup.py)
```

## Three things an agent should never break

1. **No lookahead bias** — any feature at `t` must be `f(data[..t])`; any training
   window trains on the strict past. See `03-decisions.md` and `05-testing.md`.
2. **Numba kernels with golden-value parity** — performance-critical numeric code
   is Numba `@njit`; every kernel has a parity test (1e-9/1e-10). No Cython.
3. **Doctests are tests** — every docstring example runs under
   `--doctest-modules`; a broken example fails CI.
