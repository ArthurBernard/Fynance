# 6 — Current status

A snapshot of what's done, what's in progress, and what's deliberately deferred —
so an agent doesn't re-investigate settled ground or assume a known stub is a bug.

## Done & working

- **Layered 2.x architecture** (data → features → signal → portfolio → backtest →
  metrics) composed through `typing.Protocol` seams; numpy is the lingua franca,
  PyTorch is confined to `models/`, every maillon is usable standalone and
  `strategy.Strategy` is an optional orchestrator.
- **Core**: `PriceSeries` (thin numpy-backed value object — composition, not
  `ndarray` subclassing; price↔return identities, numpy/torch bridges, `.pipe`)
  and the pipeline protocols (`DataSource`/`FeatureTransform`/`SignalModel`/
  `Allocator`/`CostModel`/`Metric`).
- **Data**: `load()` dispatcher + CSV/Parquet adapters, causal `align`/`resample`,
  and no-lookahead `train_test_split`/`walk_forward` (embargo/purge).
- **Features**: technical indicators (RSI/MACD/Bollinger/CCI/HMA/ROC/realized-vol/
  skew/kurt/autocorr), momentums (SMA/EMA/WMA + std variants), scaling (z-score,
  rolling rank), statistics, feature engineering (multi-resolution, Granger,
  incremental moments) and k-means market-regime detection.
- **Metrics**: `fynance.metrics` (Sharpe/Sortino/Calmar/diversified-ratio,
  annual return/vol, drawdown/mdd, perf_*, roll_*) + one-call `summary`.
- **Models**: econometric (ARMA/GARCH) + neural (MLP, RNN/GRU/LSTM, attention,
  TCN, Transformer) on PyTorch; `StackingEnsemble` (direction+magnitude OOF
  meta-model); differentiable losses (Sharpe/Sortino/Calmar/Omega/directional/
  hybrid) in `models/loss/`; robust-training utils (purged CV, early stopping,
  sample weighting) in `models/training.py`. All NN models conform to
  `SignalModel` (`fit`/`predict`).
- **Signal / Portfolio**: `signal/` mappers (`sign`/`threshold`/`rank`/
  vol-targeting + `SignalPipeline`); `portfolio/` allocation (ERC/HRP/IVP/MDP/MVP)
  + sizing (fractional Kelly, vol-targeting, transaction costs).
- **Backtest / Plot**: vectorized `backtest()` engine → `BacktestResult`,
  `ProportionalCost`; reporting via `fynance.plot` (`tearsheet`, composable
  figures, lazy matplotlib so `import fynance` stays matplotlib-free).
- **Research harness** (`fynance.research`, S1–S3 complete): `Experiment`
  (serializable spec + code + seed + metrics + curves), `run_experiment` (seeded,
  cost-aware, walk-forward; no-lookahead probe), `write_report` (portable markdown +
  tearsheet PNG + notebook), `gbm`/`regime_switching` synthetic generators,
  **guardrails** (`permutation_test`, `probabilistic_sharpe_ratio`,
  `deflated_sharpe_ratio`), comparison report (`compare_report`/`leaderboard`) and a
  persistent `Ledger` (append/load/leaderboard, `n_trials`, deflated-Sharpe vs
  trials). **Data-agnostic and result-free**: artifacts only go to a caller-provided
  `output_dir`; real data + result storage live in a separate private repo. Driven
  by the user-level `/run-strategy` skill.
- **Numerical kernels on Numba `@njit`** — no Cython anywhere; the build is
  pure-Python (no `setup.py`, no compile step). Every kernel has a golden-value
  parity test (1e-9/1e-10) captured from the former Cython.
- **Quality gates (all 4 enforced in CI)**: 501 unit tests + doctests on every
  module (`--doctest-modules`), incl. property tests (kernel parity + no-lookahead);
  `ruff`; `interrogate` (docstring coverage ≥ 80%, currently ~93.6%); Sphinx
  build `-W`; **`mypy` clean (0 errors)**.
- **Released**: `v2.1.1` on `master` + PyPI; `Production/Stable`. CI matrix
  3.10–3.13; release builds a pure-Python universal wheel + sdist and creates the
  GitHub Release from the CHANGELOG on tag.

## In progress / active surface

- **R&D** (`models/`): empirical comparison of losses / architectures / feature
  normalization on real out-of-sample data, market-regime conditioning, and
  multi-series OHLCV indicators — see the roadmap (§1). Most items are blocked on
  having a real dataset / orchestration rather than on missing code.
- **Research harness** (roadmap §2): S1–S3 shipped (above). Optional next: a
  Streamlit explorer over the ledger; real-data adapters + result storage live in
  the separate private research repo, not here.

## Known gaps / sharp edges (by design or deferred)

- **`estimator.estimation()`** is an explicit experimental stub: it raises
  `NotImplementedError` and points to `models.econometric_models.get_parameters`
  (the Numba-backed authoritative path).
- **Legacy `backtest` plot stack** (`PlotBackTest`/`DynaPlotBackTest`/
  `display_perf`) still powers `RollMultiLayerPerceptron` live-training viz; it is
  off the eager public surface (lazy-imported submodules) and conceptually
  superseded by `fynance.plot` — a candidate for retirement onto `fynance.plot`.
- **Notebooks** (`Notebooks/`): `quickstart_v2.ipynb` is the current tour; older
  Keras-era notebooks may remain and are not maintained.
- **`# type: ignore`** markers exist for genuinely-unmodellable cases (torch
  multiple-inheritance mixins, decorator-filled `w`); `warn_return_any` is off
  (numpy returns `Any` pervasively).

## Tooling & process

- **Dev loop**: tooled by user-level skills (`/pick-task → /plan → /execute-leaf
  → /finish-task → /release`, plus `/abandon-task`, `/groom-docs`). Docs of record
  wired in `.claude/workflow.json` (`roadmap`/`decisions`/`status`/`plans_dir`).
- **Tracked docs**: the descriptive pack `01–07` (including the roadmap),
  `README.md`, `plans/README.md` and `CLAUDE.md` are tracked. Only the plan trees
  (`doc/dev/plans/<epic>/`), any `_archive` snapshot and the Claude harness
  settings (`.claude/`) stay local.
- **Git Flow**: `master`/`develop` + `feat/fix/chore/docs` branches, one PR per
  concern (see `CLAUDE.md`).

## Deferred

Larger axes parked for later: realistic backtesting beyond proportional cost
(non-linear slippage / market impact), market-regime conditioning of the
architecture, adaptive windows, and multi-series OHLCV indicators (ATR/ADX/OBV/
VWAP) requiring a multi-series input API. Tracked in the roadmap; not bugs.

## 2026-06-16 — fynance 2.1.x shipped; post-release audit clean

The full 2.0 + 2.1 refactor is **complete and released**: 2.0 (E1–E10 — layered
architecture, `PriceSeries`, protocols, data adapters, metrics extraction,
vectorized backtest, `tearsheet`, signal/portfolio, `Strategy`, Streamlit
playground, Sphinx/notebook/README/migration docs), 2.1.0 (E7 — all Cython ported
to Numba, pure-Python build, `SignalModel` conformance), 2.1.1 (O(n) rolling
extrema perf pass + allocation cleanup). A full object-by-object audit (501 tests,
4 gates green) found one real correctness bug — `ARMAX_GARCH` swapped its `psi`/
`theta` coefficients (long-standing, preserved verbatim through the Cython→Numba
port) — plus minor dead code, a leftover debug print, and eager matplotlib import.
All fixed on `develop`. Remaining open work is the empirical R&D (§1), blocked on
data rather than code.
