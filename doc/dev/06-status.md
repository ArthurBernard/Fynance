# 6 — Current status

A snapshot of what's done, what's in progress, and what's deliberately deferred —
so an agent doesn't re-investigate settled ground or assume a known stub is a bug.

## Done & working

- **Stable numerical core**: `features` (metrics/momentums/filters/scale/roll) with
  Cython kernels + Python wrappers; `algorithms.allocation` (ERC/HRP/IVP/MDP/MVP)
  with a stable public API; `estimator` (Cython ARMA/GARCH).
- **`features/metrics.py` split** by concern into `returns.py`, `ratios.py`,
  `drawdown.py`, `stats.py` + `_metrics_helpers.py`; `metrics.py` kept as a thin
  re-export aggregator (public API + import paths unchanged).
- **Walk-forward base**: `_RollingBasis` iterator + `RollMultiLayerPerceptron` and
  `rolling_allocation()` — the causal windowing used everywhere.
- **Models**: econometric (ARMA/GARCH) + neural (MLP, RNN/GRU/LSTM, attention,
  **TCN**, **Transformer**) on PyTorch; a **StackingEnsemble** (direction+magnitude
  OOF meta-model); custom losses (Sharpe/Sortino/Calmar/Omega/directional/hybrid)
  in `models/loss/`; robust-training utils (purged CV, early stopping, sample
  weighting) in `models/training.py`.
- **Data frames**: **polars** at the input edges (`set_data`, `MA`), **numpy** at
  the output edges (`rolling_allocation`, `get_stats`); pandas removed.
- **Backtest**: static + dynamic plotting (orchestrator `BacktestNeuralNet` split
  into `backtest/backtest_neural_net.py`), perf/stat printing.
- **Quality gates (all 4 enforced in CI)**: ~300 unit tests + doctests on every
  module (`--doctest-modules`) incl. property tests (kernel parity + no-lookahead);
  `ruff`; `interrogate` (docstring coverage ≥ 80%); Sphinx build `-W`; **`mypy`
  clean (0 errors)**.
- **Released**: `v1.3.4` on `master` + PyPI; `Production/Stable`. CI matrix
  3.10–3.13; release builds manylinux wheels + sdist and creates the GitHub
  Release from the CHANGELOG on tag.

## In progress / active surface

- **ML modernisation** (`models/`): Keras/TensorFlow fully retired (no TF/Keras in
  the package); new architectures (TCN, Transformer) and custom multi-objective
  losses are the active R&D axis. See the roadmap (§1, §5).

## Known gaps / sharp edges (by design or deferred)

- **`estimator.estimation()`** is an explicit experimental stub: it raises
  `NotImplementedError` and points to `models.econometric_models.get_parameters`
  (the Cython-backed authoritative path).
- **Notebooks** (`Notebooks/`) still carry Keras examples — to be rewritten in
  PyTorch (roadmap §3.2).
- **`# type: ignore`** markers exist for genuinely-unmodellable cases (torch
  multiple-inheritance mixins, decorator-filled `w`, star-imported Cython names);
  `warn_return_any` is off (numpy/Cython return `Any` pervasively).

## Tooling & process

- **Dev loop**: tooled by user-level skills (`/pick-task → /plan → /execute-leaf
  → /finish-task → /release`, plus `/abandon-task`, `/groom-docs`). Docs of record
  wired in `.claude/workflow.json` (`roadmap`/`decisions`/`status`/`plans_dir`).
- **Tracked docs**: the full descriptive pack `01–07` (including the roadmap),
  `README.md`, `plans/README.md` and `CLAUDE.md` are tracked, mirroring dccd. Only
  the plan trees (`doc/dev/plans/<epic>/`), any `_archive` snapshot and the Claude
  harness settings (`.claude/`) stay local.
- **Git Flow**: `master`/`develop` + `feat/fix/chore/docs` branches, one PR per
  concern (see `CLAUDE.md`).

## Deferred

Larger axes parked for later: realistic backtesting (transaction costs, position
sizing), market-regime conditioning, multi-resolution features, and the
docs-hosting decision. Tracked in the roadmap; not bugs.

## 2026-06-15 — fynance 2.0 refactor: 9/10 epics shipped

The 2.0 architecture is **complete and releasable**. Shipped on `develop` (PRs
#99–#108): E1 core (`PriceSeries`, protocols), E2 data (adapters, splits), E3
metrics extraction, E4 vectorized backtest engine, E5 reporting (`tearsheet`),
E6 signal + portfolio rename, E8 `Strategy`, E9 Streamlit playground, E10 docs
(Sphinx, notebook, README, `MIGRATION-2.0.md`). Top-level API re-exports all
maillons. 462 tests, 4 CI gates green per PR.

**Remaining**: E7 (Cython→numba: estimator/econometric port, drop Cython build,
`SignalModel` conformance). Deliberately deferred — correctness-critical
numerical port (best done with golden-value parity), non-blocking for v2.0.0.
Decision pending: do E7 before the release, or cut **v2.0.0** now and land numba
in 2.1.
