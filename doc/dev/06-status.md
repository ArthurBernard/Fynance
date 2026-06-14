# 6 — Current status

A snapshot of what's done, what's in progress, and what's deliberately deferred —
so an agent doesn't re-investigate settled ground or assume a known stub is a bug.

## Done & working

- **Stable numerical core**: `features` (metrics/momentums/filters/scale/roll) with
  Cython kernels + Python twins; `algorithms.allocation` (ERC/HRP/IVP/MDP/MVP) with
  a stable public API; `estimator` (Cython ARMA/GARCH).
- **Walk-forward base**: `_RollingBasis` iterator + `RollMultiLayerPerceptron` and
  `rolling_allocation()` — the causal windowing used everywhere.
- **Models**: econometric (ARMA/GARCH) + neural (MLP, RNN/GRU/LSTM, attention) on
  PyTorch; custom financial losses (Sharpe/Sortino/directional) in `models/loss/`.
- **Backtest**: static + dynamic plotting, perf/stat printing.
- **Quality gates**: ~250 unit tests + doctests on every module
  (`--doctest-modules`); `ruff`, `mypy`, `pytest-benchmark`; Sphinx (furo) docs.
- **Released**: `v1.3.4` on `master` + PyPI; `Production/Stable`. CI matrix
  3.10–3.13; release builds manylinux wheels + sdist and (now) creates the GitHub
  Release from the CHANGELOG on tag.

## In progress / active surface

- **ML modernisation** (`models/`): migrating off Keras/TensorFlow to PyTorch;
  new architectures (TCN, Transformer) and custom multi-objective losses are the
  active R&D axis. See the (local) roadmap.
- **Structural refactors**: splitting the large modules (`features/metrics.py`,
  several `models/` and `backtest/` files). Tracked, not started.

## Known gaps / sharp edges (by design or deferred)

- **Dead stubs awaiting removal**: `models/basis.py`
  (`SignalModel`/`MagnitudeModel`), the deprecated `__BacktestNeuralNet` and the
  unused `_BacktestNeuralNet` in `backtest/dynamic_plot_backtest.py`.
- **Large modules** not yet split (see `04-subpackages.md`).
- **Notebooks** (`Notebooks/`) still carry Keras examples — to be rewritten in
  PyTorch.
- **Docs hosting**: GitHub Pages today; RTD migration is an open question.

## Tooling & process

- **Dev loop**: tooled by user-level skills (`/pick-task → /plan → /execute-leaf
  → /finish-task → /release`, plus `/abandon-task`, `/groom-docs`). Docs of record
  wired in `.claude/workflow.json` (`roadmap`/`decisions`/`status`/`plans_dir`).
- **Privacy posture**: the full descriptive pack `01–07` (including the
  roadmap), `README.md`, `plans/README.md` **and `CLAUDE.md`** are tracked,
  mirroring dccd. Only the plan trees (`doc/dev/plans/<epic>/`), any `_archive`
  snapshot and the Claude harness settings (`.claude/`) stay local.
- **Git Flow**: `master`/`develop` + `feat/fix/chore/docs` branches, one PR per
  concern (see `CLAUDE.md`).

## Deferred

Larger axes parked until the active modernisation lands: realistic backtesting
(transaction costs, position sizing), market-regime conditioning, multi-resolution
features, and the docs-hosting decision. Not started; do not treat as bugs.
