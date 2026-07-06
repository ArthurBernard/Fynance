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
  `Allocator`/`CostModel`/`Metric`); `core.checks` (`check_conforms` +
  the `assert_causal` lookahead probe) and duck-typed `from_pandas`/
  `to_pandas`/`to_polars` seams on `PriceSeries`/`OHLCV`/`BacktestResult`.
- **Data**: `load()` dispatcher + CSV/Parquet adapters, causal `align`/`resample`,
  and no-lookahead `train_test_split`/`walk_forward` (embargo/purge).
- **Features**: technical indicators (RSI/MACD/Bollinger/CCI/HMA/ROC/realized-vol/
  skew/kurt/autocorr), **OHLCV indicators** (ATR/ADX/Williams %R/OBV/VWAP),
  **causal GARCH(1,1) conditional volatility** as a feature (`garch_volatility`),
  momentums (SMA/EMA/WMA + std variants), **adaptive windows** (`adaptive_roll`/
  `adaptive_volatility`), scaling (z-score, rolling rank), statistics, money
  management (`iso_vol`), feature engineering (multi-resolution, Granger,
  incremental moments), k-means market-regime detection, and the
  **factor-research bricks (2026-07, PRs #243–#248)**: NaN-aware
  cross-sectional operators (`cs_rank`/`cs_zscore`/`cs_demean`/
  `cs_winsorize`/`cs_neutralize`), pairwise rolling stats
  (`roll_cov`/`roll_corr`/`roll_beta`, `cross_corr`), fixed-width
  fractional differentiation (`fracdiff`), and the AFML labeling stack
  (`triple_barrier`/`meta_labels`/`uniqueness_weights` — training targets
  for purged splits).
- **Metrics**: `fynance.metrics` (Sharpe/Sortino/Calmar/diversified-ratio,
  annual return/vol, drawdown/mdd, perf_*, roll_*, **`information_coefficient`**
  rank-IC) + one-call `summary`; **2026-07 additions (PRs #253–#257)**:
  benchmark-relative family (`benchmark` — alpha/beta/TE/IR/capture),
  tail risk (`risk` — VaR/CVaR/CDaR + `tail_dependence`), turnover/exposure
  analytics and round-trip **trade analytics** (`trades` +
  `BacktestResult.trades()`).
- **Multi-asset / panel harness**: `ObjectiveModel` trains a position book
  `(T, N)` from a panel `X`; book-aware ratio losses + `RankingLoss`;
  `Strategy`/`run_walk_forward`/`run_experiment` accept a `(T, N)` panel and return
  a book equity with per-asset attribution (`BacktestResult.asset_gross_returns`);
  `horizon_returns` non-overlapping labels; book tearsheet (per-asset contribution
  + turnover). The single-asset `N=1` path is unchanged.
- **Models**: econometric (ARMA/GARCH) + neural (MLP, RNN/GRU/LSTM, attention,
  TCN, Transformer) on PyTorch; `StackingEnsemble` (direction+magnitude OOF
  meta-model); **`RegimeMoE`** (regime-conditioned mixture-of-experts — an
  objective-aligned net gated on a **causal** regime label); differentiable
  losses (Sharpe/Sortino/Calmar/Omega/directional/hybrid) in `models/loss/`;
  robust-training utils (purged CV, early stopping, sample weighting) in
  `models/training.py`. All NN models conform to `SignalModel` (`fit`/`predict`).
  **2026-07 ML bricks (PRs #263–#266)**: cross-asset pretraining + save/load on
  `ObjectiveModel` (`pretrain_pooled`/`clone`/`finetune`), distributional
  `QuantileModel` (+ `PinballLoss`), uncertainty wrappers (`DeepEnsemble`,
  `MCDropout`) and causal split-`conformal` intervals; **GARCH family (PRs
  #255/#262)**: GJR/EGARCH kernels + Student-t + `fit_volatility` MLE driver.
- **Objective-aligned training** (`models/objective.py`): `ObjectiveModel` trains
  any `nn.Module` (MLP by default) **directly on a differentiable financial
  objective** (`SharpeLoss`/`SortinoLoss`/…) — the net outputs positions, the loss
  is computed on `positions * returns`. **Net-of-cost** (`cost=` penalizes
  turnover so the net learns to *hold*) and **mini-batch** (`batch_size`/`shuffle`,
  contiguous chunks — what makes it converge on long minute-resolution series).
  Plugs into the harness via the `X` path with `signal=identity`, `y=returns`.
- **Signal / Portfolio**: `signal/` mappers (`sign`/`threshold`/`rank`/
  vol-targeting + **anti-churn** `ema_smooth`/`deadband`/`min_hold` +
  `SignalPipeline`); `portfolio/` allocation (ERC/HRP/IVP/MDP/MVP + **`RBP`**
  risk budgeting) + sizing (fractional Kelly, vol-targeting — single-series
  `vol_target` and book-level **`book_vol_target`** — transaction costs);
  **portfolio-risk bricks (2026-07, PRs #235–#240)**: conditioned covariance
  estimators (`covariance` — Ledoit-Wolf/EWMA/factor/Marchenko-Pastur) behind
  an opt-in `cov=` seam on every allocator, ex-ante/rolling risk
  **attribution**, and a least-distance exposure-**constraints** overlay
  (`project_weights`).
- **Backtest / Plot**: vectorized `backtest()` engine → `BacktestResult`, cost
  models (`ProportionalCost` + **`MarketImpactCost`** + **`HoldingCost`** /
  **`CompositeCost`** — borrow/financing/cash carry, composable stacking);
  **rebalancing policies** (`portfolio.rebalance` — calendar/band/turnover-cap +
  `discretize`/`delay`), **capacity analysis** (`capacity_curve`/`breakeven_fee`)
  and **intraday session utilities** (`data.sessions`); reporting via
  `fynance.plot` (`tearsheet`, composable figures, lazy matplotlib so
  `import fynance` stays matplotlib-free).
- **Research harness** (`fynance.research`, S1–S3 complete): `Experiment`
  (serializable spec + code + seed + metrics + curves), `run_experiment` (seeded,
  cost-aware, walk-forward; no-lookahead probe; records a **provenance** block —
  data/features/model/run config — into the spec), `write_report` (portable
  markdown with a Provenance table + tearsheet PNG + notebook),
  `gbm`/`regime_switching` synthetic generators,
  **guardrails** (`permutation_test`, `probabilistic_sharpe_ratio`,
  `deflated_sharpe_ratio`), comparison report (`compare_report`/`leaderboard`) and a
  persistent `Ledger` (append/load/leaderboard, `n_trials`, deflated-Sharpe vs
  trials). **Multi-input**: `Strategy`/`run_experiment` accept a precomputed
  feature matrix `X`/`y` (price → P&L only; walk-forward slices `X` per window =
  the rolling-NN refit), and `fynance.features.RegimeDetector` gives a **causal**
  regime label (fit-on-train/assign-online; `detect_regimes` stays in-sample for
  analysis). **Data-agnostic and result-free**: artifacts only go to a
  caller-provided `output_dir`; real data + result storage live in a separate
  private repo (`fynance-research`). Driven by the user-level `/run-strategy` skill.
- **Numerical kernels on Numba `@njit`** — no Cython anywhere; the build is
  pure-Python (no `setup.py`, no compile step). Every kernel has a golden-value
  parity test (1e-9/1e-10) captured from the former Cython.
- **Quality gates (all 4 enforced in CI)**: ~880 tests collected (~790 unit/
  benchmark + doctests on every module via `--doctest-modules`), incl. property
  tests (kernel parity + no-lookahead); `ruff`; `interrogate` (docstring coverage
  ≥ 80%, currently ~94.7%); Sphinx build `-W`; **`mypy` clean (0 errors)**.
- **Released**: `v2.10.1` on `master` + PyPI; `Production/Stable`. CI matrix
  3.10–3.13; release builds a pure-Python universal wheel + sdist and creates the
  GitHub Release from the CHANGELOG on tag.

## In progress / active surface

The **2026-07 feature backlog** (roadmap §2–§10, from the judge-scored
ideation catalog) is being executed epic by epic; `portfolio-risk` (§3) is
**done** (PRs #235–#240, released as **v2.12.0**) and `factor-research` (§4)
is **done** (PRs #243–#248: cross-sectional ops, pairwise rolling stats,
factor suite `metrics.factor`/`plot.factor`, `fracdiff`, AFML labels,
walk-forward MDA in `research.importance`), as is `anti-overfitting` (§5,
PRs #249–#252: purged walk-forward HP search with `n_trials` → deflated
Sharpe, CSCV/PBO, block/stationary bootstrap, CPCV splitter). The v2.9.0 **library
bricks** all shipped (OHLCV indicators, causal GARCH-volatility feature,
adaptive windows, `RegimeMoE`, `MarketImpactCost`), and the **2026-06 audit**
was fully remediated across two passes (v2.10.0: PRs #188–#196; v2.10.1: PRs
#201–#207 — correctness/tests/docs the CI gates don't catch, incl. one KPI
regression). The CI/badge hygiene chores are resolved (CI runs on `develop` PRs;
badges green). The only remaining open item is the optional **Streamlit ledger
explorer** (roadmap §2). See `07-roadmap.md`.

**Out of scope here**: strategy research on **real data** (empirical loss /
architecture / normalization benchmarks, out-of-sample Sharpe, online regimes,
feature selection) lives in the separate **private repo** `fynance-research`,
which depends on fynance. This public repo stays data-agnostic and result-free.

## Known gaps / sharp edges (by design or deferred)

- **`estimator.estimation()`** remains an ARMA-oriented experimental stub
  (`NotImplementedError`), now pointing to `estimator.fit_volatility` for the
  volatility-model MLE path shipped in 2026-07 (PRs #255/#262: GJR/EGARCH
  kernels + Student-t + the `fit_volatility` driver with forecast/simulate).
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

The library bricks that were parked here — realistic backtesting beyond
proportional cost (non-linear market impact), market-regime conditioning of the
architecture, adaptive windows, and multi-series OHLCV indicators
(ATR/ADX/OBV/VWAP) — **all shipped in v2.9.0**. The only library axis still
deferred is the optional **Streamlit ledger explorer** (roadmap §2). Tracked in
the roadmap; not bugs.

## 2026-06-22 — library bricks shipped (v2.9.0); audit remediation opened

The data-agnostic **library bricks** that the re-scoped roadmap kept all landed
in **v2.9.0**: an `OHLCV` container + multi-series indicators
(ATR/ADX/Williams %R/OBV/VWAP), the **causal GARCH(1,1) conditional-volatility
feature** (`garch_volatility`), **adaptive windows** (`adaptive_roll`/
`adaptive_volatility`), regime-conditioned architecture (`RegimeMoE`), and a
non-linear **`MarketImpactCost`**. A full audit (2026-06-22) — all five gates
(pytest/ruff/mypy/interrogate/sphinx) already green — surfaced correctness bugs,
test gaps and doc drift the gates don't catch; tracked as the parallelizable
remediation backlog in roadmap §1.

## 2026-06-21 — research line shipped (v2.2 → v2.8); roadmap re-scoped

Since 2.1.1 the **R&D enablement line** shipped end-to-end: the `fynance.research`
harness (S1–S3 — `Experiment`/`run_experiment`/`write_report`, synthetic
generators, permutation / deflated-Sharpe guardrails, `Ledger`/leaderboard,
multi-input `X`/`y`, causal `RegimeDetector`, self-describing provenance) and the
**objective-aligned training brick** `ObjectiveModel` (v2.5 differentiable
objective → v2.7 net-of-cost turnover penalty + anti-churn signal mappers → v2.8
mini-batch training that makes it converge on long series). The roadmap was
**re-scoped**: strategy research on real data moves to the private
`fynance-research` repo; the public roadmap keeps only data-agnostic library
bricks (multi-series OHLCV indicators, regime-conditioned architecture, adaptive
windows, realistic backtest, Streamlit explorer). Latest release **v2.8.0**.

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
