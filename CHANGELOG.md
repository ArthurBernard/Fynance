# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking Changes

### Added

- **Tail-risk metrics.** New `fynance.metrics.risk`: `var` / `cvar`
  (historical, Gaussian, Cornish-Fisher), `cdar` (mean of the α worst
  drawdowns), causal `roll_var`/`roll_cvar`, and pairwise lower-tail
  `tail_dependence` on `(T, N)` panels; scalar metrics registered in
  `summary()`.
- **Benchmark-relative metrics.** New `fynance.metrics.benchmark`:
  `beta`, `alpha` (annualized Jensen), `tracking_error`,
  `information_ratio`, up/down `capture_ratio`, `benchmark_summary` and
  `roll_beta_benchmark` (reusing the `features.roll_functions` kernel).
- **Block / stationary bootstrap.** `fynance.research.bootstrap`:
  `resample_paths` (circular + Politis-Romano stationary, Numba kernels),
  `bootstrap_metric` (percentile CIs on any metric) and
  `block_permutation_test` — a dependence-preserving null complementing
  `guards.permutation_test`.
- **Probability of backtest overfitting.** `pbo` in
  `fynance.research.overfit`: CSCV diagnostic over a `(T, n_configs)`
  returns panel (rank logits, `prob_oos_loss`, IS→OOS degradation slope),
  plus `returns_panel()` to build the input from `Experiment` objects.
- **Combinatorial purged cross-validation.** `combinatorial_purged_cv` in
  `fynance.data.split`: every C(n_groups, n_test_groups) group combination
  becomes an out-of-sample path, with purge windows around test-group
  boundaries and post-test embargo (AFML) — many OOS paths instead of one.
- **Purged walk-forward hyperparameter search.** `walk_forward_search` in
  `fynance.models.tuning`: grid/random search evaluated out-of-fold on the
  purged walk-forward splitter, returning a `SearchResult` whose `n_trials`
  feeds `deflated_sharpe_ratio` — trial accounting honest by construction.

- **AFML labeling stack.** New `fynance.features.labels` module:
  `triple_barrier` (path-aware profit-take / stop-loss / vertical-barrier
  labels on a Numba scan, volatility-scaled barriers), `meta_labels`
  (did-the-bet-pay binary target), `label_concurrency` and
  `uniqueness_weights` (overlap-aware sample weights). Documented as
  training TARGETS (they read future prices by design) to be consumed only
  through purged splits.
- **Factor analysis suite.** New `fynance.metrics.factor`
  (`quantile_returns` -> `QuantileResult`, `roll_information_coefficient`,
  `ic_decay` over non-overlapping horizons, `ic_summary` with ICIR/t-stat/
  hit-rate, `factor_rank_autocorr`) and `fynance.plot.factor`
  (`plot_quantile_returns`, `plot_ic_series`, `plot_ic_decay` and the
  composed 2x2 `factor_tearsheet`) — alphalens-style factor evaluation on
  data-agnostic `(T, N)` panels.
- **Walk-forward feature importance (MDA).** `walk_forward_mda` in
  `fynance.research.importance`: permutation importance evaluated
  out-of-fold on the purged walk-forward splitter — fit once per fold,
  permute the test window only, seeded; returns an `ImportanceResult`
  (per-feature mean/std score drop + baseline).
- **Fractional differentiation.** `fracdiff(X, d, tol)` in
  `fynance.features.engineering`: fixed-width-window fractional
  differentiation (AFML ch. 5) on a Numba kernel — stationarize while
  keeping maximal memory; strictly causal, NaN warm-up head.
- **Cross-sectional operators.** New `fynance.features.cross_section`
  module: NaN-aware per-bar panel transforms `cs_rank` (average-tie
  percentile ranks), `cs_zscore`, `cs_demean` (optionally weighted),
  `cs_winsorize` and `cs_neutralize` (per-bar OLS residualization against
  exposure panels) — the input half of the `(T, N)` factor workflow.
- **Pairwise rolling statistics.** `roll_cov` / `roll_corr` / `roll_beta`
  (trailing two-series moments on Numba kernels, window inclusive of `t`)
  and `cross_corr` (lead-lag correlation profile) in
  `fynance.features.roll_functions`.

### Changed

### Fixed

### Deprecated

### Removed

## [2.12.0] - 2026-07-05

### Added

- **Book-level vol targeting.** `book_vol_target(W, X, ...)` in
  `fynance.portfolio.sizing`: causal `(T,)` leverage series targeting a
  constant volatility for a whole `(T, N)` position book (weights decided at
  `t-1` earn the return over `(t-1, t]`) — the multi-asset counterpart of
  `vol_target`.
- **Exposure-constraint overlay.** New `fynance.portfolio.constraints`
  module: `project_weights` projects a weight vector or `(T, N)` book onto a
  feasible set — per-asset box, gross-leverage cap, net-exposure range, named
  group bounds — as a least-distance SLSQP projection (split `v = p - m`
  formulation), with a fast clip-and-scale path for box+gross and an
  interval-arithmetic infeasibility pre-check.
- **Risk-budgeting allocation.** New `RBP(X, budgets=None, ...)` allocator
  (generalized ERC, Roncalli least-squares objective): weights whose risk
  contributions match an arbitrary budget vector; `budgets=None` reproduces
  `ERC`, and the `cov=` seam / `rolling_allocation` forwarding apply.
- **Risk attribution.** New `fynance.portfolio.attribution` module:
  `marginal_risk` (∂σₚ/∂w), `risk_contribution` (absolute or percentage,
  summing to σₚ / 1) and the causal `roll_risk_contribution` walking a
  `(T, N)` weight path against a trailing covariance window (accepts the
  same `cov=` estimators as the allocators).
- **Allocator `cov=` seam.** The six allocators (`ERC`/`HRP`/`IVP`/`MVP`/
  `MVP_uc`/`MDP`) accept an opt-in `cov=` callable mapping the `(T, N)`
  training window to an `(N, N)` covariance matrix (e.g.
  `portfolio.covariance.ledoit_wolf`); the default `None` keeps the
  sample-covariance path bit-for-bit, and `rolling_allocation` forwards
  `cov=` through its existing `**kwargs`.
- **Conditioned covariance estimators.** New `fynance.portfolio.covariance`
  module: `sample_cov`, closed-form `ledoit_wolf` shrinkage (identity /
  constant-correlation / diagonal targets), RiskMetrics-style `ewma_cov`
  (Numba kernel), PCA `factor_cov` (low-rank + diagonal, PSD by construction)
  and Marchenko-Pastur `denoise_cov` — interchangeable `(T, N) → (N, N)`
  callables, groundwork for the allocators' opt-in `cov=` seam.

### Changed

### Fixed

### Deprecated

### Removed

## [2.11.2] - 2026-06-25

### Added

- **Trading-profile metrics.** `sign_changes` / `trades_per_year`
  (`fynance.metrics`) count position direction flips (long↔flat↔short), total
  and per-asset for a book — the round-trip churn a turnover-blind `total_cost`
  hides. `BacktestResult.summary()` now reports `n_sign_changes` and
  `trades_per_year`, so they flow into the research report automatically.
- **Cost decomposition.** Cost models may expose an optional
  `components(weights)` breakdown (`ProportionalCost` → `transaction`;
  `MarketImpactCost` → `transaction` + `market_impact`); the engine carries it
  on `BacktestResult.cost_components`, the research runner persists it, and the
  tearsheet stacks a full-width **cumulative-fees** panel (cost in % of capital
  by source) via the new `plot_cost_decomposition`.

### Changed

### Fixed

### Deprecated

### Removed

## [2.11.1] - 2026-06-25

### Added

- **Tearsheet equity readability.** `plot_equity` / `tearsheet` gained two
  display-only options: `base` (rescale the curve to start at e.g. `100` for a
  base-100 reading) and `logy` (`"auto"` switches to a log y-axis on
  wide-amplitude curves — a x3-x30 trajectory stays readable and drawdowns stay
  comparable across time — overridable with `True`/`False`).

### Changed

### Fixed

### Deprecated

### Removed

## [2.11.0] - 2026-06-23

### Breaking Changes

### Added

- **Multi-asset / panel research harness.** `ObjectiveModel` trains a position
  **book** `(T, N)` from a panel `X` `(T, N, M)` (or `(T, N·M)`) with a per-asset
  target `y` `(T, N)`, scored on the aggregated book objective; `Strategy.run`/
  `run_walk_forward` and `run_experiment` accept a `(T, N)` panel, stitch a
  per-asset book and return a book equity with per-asset attribution
  (`BacktestResult.asset_gross_returns`); the book tearsheet adds per-asset
  **contribution** and **turnover** panels. The single-asset `N=1` path is
  numerically unchanged throughout.
- **`information_coefficient`** (rank-IC / Pearson; cross-sectional per bar or
  per-asset time-series) and **`horizon_returns`** (non-overlapping forward
  labels) — a predict-then-rule guardrail to gauge signal quality before trading.
- **`RankingLoss`** — a differentiable cross-sectional long-short ranking loss.

### Changed

- **Ratio losses are book-aware.** `SharpeLoss`/`SortinoLoss`/`CalmarLoss`/
  `OmegaLoss` aggregate a 2-D `(T, N)` position book to the 1-D book return
  (`Σ_i posᵢ·rᵢ`) before scoring; 1-D and `(T, 1)` inputs are numerically
  unchanged.

### Fixed

### Deprecated

### Removed

## [2.10.2] - 2026-06-23

### Breaking Changes

### Added

### Changed

- **Research reports plot against dates.** When the `PriceSeries` passed to
  `run_experiment` carries a `datetime64` index, the experiment now persists a
  tail-aligned date index and `write_report`'s tearsheet draws equity, drawdown
  and rolling Sharpe against real dates instead of bar numbers (it falls back to
  bar numbers when no temporal index is present).

### Fixed

### Deprecated

### Removed

## [2.10.1] - 2026-06-22

### Breaking Changes

### Added

### Changed

- **`RNN`/`GRU`/`LSTM` honor the `SignalModel` contract.** `fit(X, y)` and
  `predict(X)` now work with a zero-initialized hidden (and cell) state — the
  natural default for these stateless gated cells. The explicit-state forms
  (`train_on(X, y, H[, C])`, `predict(X, H[, C])`) are unchanged.
- **Custom losses use a smooth saturating map.** `CalmarLoss`/`OmegaLoss`/
  `SortinoLoss` replace the hard `MAX_RATIO` clamp (which zeroed the gradient on
  low-risk batches) with `MAX_RATIO * tanh(ratio / MAX_RATIO)` plus a
  scale-invariant floor — finite loss, gradient preserved, normal-regime
  numerics unchanged.

### Fixed

- **`roll_standardize`/`roll_normalize` with `axis=1`** on genuine multi-column
  input (the 2.10.0 `axis=1` repair had only covered the `Scale` class).
- **`RollMultiLayerPerceptron.run(backtest_kpi=True)`** (the default) raised
  `IndexError` on the final post-loop print; the KPI index is now clamped.
- **`_safe_ratio` returns `-inf`** (not `+inf`) for a negative excess over a
  zero denominator — a riskless loss is no longer scored as the best ratio;
  `roll_sharpe` is unified onto the same `_safe_ratio` convention.
- **`_wrappers` negative axis** (`axis=-1`) now resolves correctly instead of
  silently computing `axis=0`; `accuracy`/`directional_accuracy` work on 2-D
  `axis=1`.
- **`data.split.walk_forward(step<=0)`** infinite loop and `train_test_split`
  negative `test_size` (out-of-bounds indices) now raise; `align.resample`
  handles `datetime64` resolutions beyond `[D]/[ms]/[us]/[ns]`; `align` rejects a
  duplicate index; `PriceSeries.to_returns(dropna=False)`/`pnl()` handle empty
  input.
- **Portfolio `ERC`/`MVP_uc` clamp `low_bound`** (feasible, sum-to-one weights);
  `research.synthetic.gbm`/`regime_switching` reject `n < 1`;
  `research.compare` leaderboard unions metric columns across rows;
  `features.money_management.iso_vol` accepts list input.

### Deprecated

### Removed

## [2.10.0] - 2026-06-22

### Breaking Changes

### Added

### Changed

- **`RNN`/`GRU`/`LSTM` are documented as stateless gated feed-forward cells**,
  not temporal recurrent nets — they process each timestep independently and do
  not thread state across a time axis. The unsatisfiable `(T, S, N)` sequence
  contract was removed from the docstrings, and the default `forward_activation`
  is now `Identity` (was `Softmax`, which forced regression outputs onto a
  probability simplex). `bias=False` is now honored.
- **`model.predict()` runs in eval mode** (dropout disabled, deterministic) and
  `set_data`/`fit` honor the requested `dtype`; float64 numpy input no longer
  crashes against float32 parameters.
- **`calmar`/`roll_calmar` return `+inf`** (not `0.0`) for a profitable
  drawdown-free curve, consistent with `sharpe`/`sortino`.
- **`features.money_management.iso_vol`** now uses the standard return
  `s_t / s_{t-1} - 1` (was the reciprocal `s_{t-1} / s_t - 1`).
- **`research.Ledger`** is strictly append-only (`append` raises on a duplicate
  name) and `Ledger.deflated_sharpe` de-annualizes the Sharpe before the DSR so
  the guard is no longer saturated.

### Fixed

- **2-D `axis=1` path across `fynance.features`.** `Scale.scale`/`Scale.revert`
  silently ignored `axis=1` (missing `return`); the rolling-window size was
  clamped against the series axis instead of the time axis; `mad(axis=1)` raised
  a broadcast error; `np.AxisError` no longer exists under NumPy 2. All fixed,
  with genuine multi-column parity tests (the old tests used degenerate
  single-column arrays).
- **`portfolio.ERC` / `MVP_uc` returned the equal-weight starting guess.** The
  SLSQP objective fell below the default `ftol` on return-scale inputs; the
  covariance is now rescaled to unit trace so they converge. Single-asset inputs
  no longer crash the covariance-based optimizers; `IVP(normalize=True)` no
  longer distorts the inverse-variance weights.
- **Custom losses** (`CalmarLoss`/`OmegaLoss`/`SortinoLoss`) no longer explode
  to huge magnitudes on low-denominator (e.g. drawdown-free) batches.
- **`_RollingBasis` walk-forward** now rejects `roll_period > train_period`
  (which previously wrapped negative indices into future data — a lookahead
  leak); train-loss normalization, per-step KPI indexing, and fork-unsafe
  plotting in `run()` are fixed.
- **`signal.rank`** raises `ValueError` on overlapping or negative legs
  (previously produced silently non-dollar-neutral weights).
- **`econometric_models.ARMA`/`ARMA_GARCH`/`ARMAX_GARCH`** accept list inputs;
  `estimator.loglikelihood` no longer mutates its input array;
  `diversified_ratio` returns a scalar.
- **`data.align.resample`** validates the index dtype with a clear error;
  `data.split.walk_forward` rejects empty-train configs;
  `core.PriceSeries.to_returns` rejects non-positive prices for log/pct returns.
- **`backtest` transaction-cost timing** on prices input;
  `research.run_experiment` no longer mutates the caller's `strategy.cost`;
  `leaderboard` sorts a NaN metric to the bottom.
- Removed a dead `bollinger_band` deprecation warning; corrected numerous
  docstring formulas/typos and `Raises` sections.

### Deprecated

### Removed

## [2.9.0] - 2026-06-21

### Breaking Changes

### Added

- **`fynance.core.OHLCV` — aligned multi-series value object.** A thin,
  numpy-backed container for aligned Open/High/Low/Close/Volume series (the
  multi-series counterpart of `PriceSeries`): `close` required, the other four
  fields optional (accessing an absent field raises), equal length enforced at
  construction, with `from_dict`/`from_numpy`/`from_polars` bridges and
  `to_numpy()`. The input contract the new OHLCV indicators consume.
- **Multi-series OHLCV indicators (`fynance.features.ohlcv`).** Five causal
  indicators that need High/Low or Volume: `atr` (Wilder ATR), `adx` (Wilder
  ADX), `williams_r`, `obv`, and `vwap` (cumulative or rolling). Each takes the
  raw aligned arrays **or** a single `OHLCV` container; rolling loops are Numba
  `@njit` kernels.
- **Causal GARCH(1,1) volatility feature (`fynance.features.garch_volatility`).**
  Conditional volatility as a strictly causal feature: GARCH(1,1) fit by maximum
  likelihood on a training prefix (optionally refit on the expanding window every
  `refit` steps), then forward-filtered over the series; the `min_train` warmup is
  `NaN`. Reuses the single authoritative ARMA/GARCH recursion and likelihood — no
  duplicated parameter logic.
- **Regime-conditioned architecture (`fynance.models.RegimeMoE`).** A
  `SignalModel` mixture-of-experts conditioned on the **causal** market regime
  (`RegimeDetector`): `routing="soft"` concatenates a learned regime embedding to
  the features through a shared trunk (default), `routing="hard"` uses one expert
  per regime. The regime label is produced by a detector fit on the training slice
  only and assigned online. Reuses `ObjectiveModel` for the training loop.
- **Regime-adaptive rolling windows (`fynance.features.adaptive_roll` /
  `adaptive_volatility`).** Apply a trailing-window feature with a per-bar window
  chosen by the current causal regime label (short window in one regime, longer in
  another); a single regime with a constant window reduces exactly to the
  fixed-window statistic.
- **Non-linear market-impact cost (`fynance.backtest.MarketImpactCost`).** A
  `CostModel` adding a convex, super-linear market-impact term on top of the
  linear fee — `fee*turnover + impact*turnover**exponent` (default `exponent=1.5`,
  the square-root impact law). `exponent=1, impact=0` reduces exactly to
  `ProportionalCost`.

### Changed

- **Roadmap re-scoped to data-agnostic library bricks.** Strategy research on
  real data (empirical loss/architecture/normalization benchmarks, out-of-sample
  Sharpe, online regimes) moved to the private `fynance-research` repo; the public
  roadmap and dev docs now track only reusable, data-agnostic library work. The
  shipped `ObjectiveModel` epic and the §1 "library bricks" epic were removed from
  the roadmap.

### Fixed

### Deprecated

### Removed

## [2.8.0] - 2026-06-18

### Breaking Changes

### Added

- **Mini-batch training for `ObjectiveModel`.** New `batch_size` (and `shuffle`) parameters train on **contiguous** mini-batches instead of full-batch — so a `fit` does `epochs * ceil(T / batch_size)` optimizer steps instead of just `epochs`. This is what makes the objective actually converge on long (e.g. minute-resolution) series, where full-batch gave far too few updates. The turnover penalty is carried across chunk boundaries (in time order); `batch_size=None` keeps the original full-batch behaviour.

### Changed

### Fixed

### Deprecated

### Removed

## [2.7.0] - 2026-06-18

### Breaking Changes

### Added

- **Anti-churn signal mappers.** `fynance.signal` gains three causal, composable position mappers to cut turnover where transaction costs dominate: `ema_smooth` (EMA-smooth a position), `deadband` (sticky hold unless the target moves beyond a band, magnitude-preserving), and `min_hold` (enforce a minimum holding period between changes). They pair with the train-time `ObjectiveModel(cost=...)` penalty.
- **Turnover-penalized (net-of-cost) objective training.** `ObjectiveModel` gains a `cost` parameter: when non-zero the objective is computed on `positions * returns - cost * |Δpositions|`, so the network learns to **hold** positions instead of churning — the anti-churn lever for high-cost / high-frequency settings. Defaults to `0` (unchanged behaviour).

### Changed

### Fixed

### Deprecated

### Removed

## [2.6.0] - 2026-06-17

### Breaking Changes

### Added

- **Reference narrative for the new capabilities.** `strategy.rst` documents the `X`/`y` multi-input contract (precomputed causal feature matrix; walk-forward slices it per window = the model refit; dtypes preserved); `models.neural_network.rst` gains an objective-aligned training section on `ObjectiveModel`; `features.regime.rst` spells out the causal `RegimeDetector` contract; `quickstart` links to the new workflow tutorial.
- **Research workflow tutorial + runnable example.** New `doc/source/research_workflow.rst` walks the canonical loop end-to-end on synthetic data (data → causal `X`/`y` features → rule-based & objective-aligned strategies → walk-forward `run_experiment` → permutation/deflated-Sharpe guardrails → portable report) — the documented, portable form of the `/run-strategy` skill. Shipped alongside a runnable `examples/research_workflow.py` (exercised in CI so it cannot rot).
- **Self-describing experiments (provenance).** `run_experiment` now records a structured *provenance* block in `Experiment.spec` — `data` (kind, length, index span, optional `data_desc`), `features` (`X` shape, optional `feature_names`/`feature_desc`), `model`, `signal`, `walk_forward`, `cost`, `period`, `seed` — so every artifact records *what produced it*. `write_report` surfaces it as a **Provenance** table at the top of `report.md`. New optional `run_experiment` keyword args `feature_names`, `feature_desc`, `data_desc`. Backward compatible: older `experiment.json` specs still load and render (the table degrades to available fields).

### Changed

### Fixed

### Deprecated

### Removed

## [2.5.0] - 2026-06-17

### Breaking Changes

### Added

- `fynance.models.ObjectiveModel` — **objective-aligned training**: a `SignalModel` that trains a neural network (any `nn.Module`; a clean MLP by default) **directly on a differentiable financial objective** (`SharpeLoss`/`SortinoLoss`/…) rather than MSE. The net outputs positions and the loss is computed on `positions * returns`; `fit(X, y)` reads `y` as the realized returns, `predict(X)` returns positions in `[-1, 1]`. Plugs into the harness via the `X` path with `signal=identity`, `y=returns`.

### Changed

### Fixed

### Deprecated

### Removed

## [2.4.1] - 2026-06-17

### Breaking Changes

### Added

### Changed

### Fixed

- `sharpe` / `sortino` crashed (`TypeError`) on a **constant / zero-volatility** equity curve — a legitimate result for a flat ("do-nothing") strategy. They now return `0.0` when the excess return is also zero and `+inf` for a riskless positive drift, for both scalar (1-D) and array (2-D) inputs (shared `_safe_ratio` helper). `summary()` no longer crashes on a flat curve.

### Deprecated

### Removed

## [2.4.0] - 2026-06-17

### Breaking Changes

### Added

- `Strategy.run` / `Strategy.run_walk_forward` and `fynance.research.run_experiment` accept a precomputed feature matrix `X` (aligned with the price index): when given it replaces `features(prices)` and the walk-forward slices `X[train]`/`X[test]` per window (the rolling-NN refit). Prices are used only for the P&L; `X` carries exogenous / regime / multi-venue inputs the price-only featurizer cannot build. `X`/`y` dtypes are preserved (so a float32 `X` matches a float32 torch model).
- `fynance.features.RegimeDetector` — a **causal** market-regime detector (fit k-means on a training slice, assign later points to the nearest training centroid), plus `regime_features` (the causal trailing vol / mean-return matrix). Unlike `detect_regimes` (in-sample), it is safe as a backtest feature. `detect_regimes` is unchanged (refactored to share `regime_features`).

### Changed

### Fixed

### Deprecated

### Removed

## [2.3.0] - 2026-06-17

### Breaking Changes

### Added

- `fynance.research` guardrails (`guards.py`): `permutation_test` (shuffle the asset's returns, build a null distribution of a metric → p-value; detects a spurious edge / leakage) and `probabilistic_sharpe_ratio` / `deflated_sharpe_ratio` (Bailey & López de Prado — does the edge survive the number of trials?). Anti-overfitting / anti-data-snooping for AI-driven search.
- `fynance.research.compare_report(experiments, output_dir)` and `leaderboard(experiments)` — rank a set of experiments and overlay their equity curves into a portable comparison report (leaderboard markdown + overlay PNG).
- `fynance.research.Ledger(root)` — a persistent experiment store: `append`/`load`/`leaderboard`, an `n_trials` count, and `deflated_sharpe(experiment)` that judges a selected strategy against the ledger's trial count and Sharpe dispersion. The store lives entirely under the caller's `root`.

### Changed

### Fixed

### Deprecated

### Removed

## [2.2.0] - 2026-06-16

### Breaking Changes

### Added

- `fynance.research` — new data-agnostic research-harness subpackage. First piece: `Experiment`, a serializable record (spec + generated code + seed + metrics + curves + provenance) with `to_dict`/`from_dict`/`save`/`load`. Artifacts are written only to a caller-provided `output_dir` (fynance never stores results itself).
- `fynance.research` synthetic generators `gbm` and `regime_switching` — seeded price paths so the harness is testable with zero real data (and usable as a null test).
- `fynance.research.run_experiment(strategy, data, *, name, walk_forward, costs, seed, output_dir, ...)` — runs a seeded, cost-aware, walk-forward (or single) backtest through the existing maillons and returns a populated `Experiment`; saves it under `output_dir` when given. No-lookahead verified by a black-box causality probe.
- `fynance.research.write_report(experiment, output_dir, *, notebook, execute)` — renders an experiment into portable, remotely-viewable artifacts (markdown + tearsheet PNG + a re-runnable notebook) under `output_dir`. matplotlib/nbformat imported lazily; notebook execution is opt-in and degrades gracefully.

### Changed

### Fixed

### Deprecated

### Removed

## [2.1.3] - 2026-06-16

### Breaking Changes

### Added

### Changed

- Removed the last stale Cython references from the user-facing surface (package docstring, `estimator`/`econometric_models` docstrings, the API-stability policy now reads "2.x"). Internal parity-test docstrings keep accurate "former Cython" provenance notes.

### Fixed

- Read the Docs builds were failing since v2.1.0 — `.readthedocs.yaml` still ran `python setup.py build_ext` (no `setup.py` in the pure-Python build), so the hosted docs were stuck on a pre-2.0 render. Dropped the obsolete build step; RTD now builds the current docs.

### Deprecated

### Removed

## [2.1.2] - 2026-06-16

### Breaking Changes

### Added

### Changed

- `import fynance` no longer eagerly imports matplotlib/seaborn (lazy in the rolling-NN live-viz path); the legacy `backtest` plot objects (`PlotBackTest`/`DynaPlotBackTest`/`display_perf`) are no longer on the eager public surface (still importable as submodules). Plotting/reporting is `fynance.plot`.
- Documentation and packaging metadata refreshed for the pure-Python (Numba) 2.x reality: dropped the stale `Cython` PyPI classifier and "Python and Cython" project description, fixed the install instructions (no compile step), and rewrote the developer brief / Sphinx pages to the layered architecture.

### Fixed

- `ARMAX_GARCH`: the external-regressor coefficients (`psi`) and MA coefficients (`theta`) were swapped between the public wrapper and the kernel, so `psi` was applied to past residuals instead of the exogenous regressor `x` (a long-standing bug, preserved verbatim through the Cython→numba port). Fixed; `ARMA_GARCH` was unaffected.

### Deprecated

### Removed

- Dead private helpers `_roll_annual_return_py` / `_roll_annual_volatility_py` (superseded numpy fallbacks) from `fynance.features._metrics_helpers`.

## [2.1.1] - 2026-06-16

### Breaking Changes

### Added

### Changed

- Removed a leftover debug `print` in `portfolio.allocation` (singular-covariance error path now re-raises cleanly).
- Performance: rolling extrema (`roll_min`/`roll_max`) reimplemented with an O(n) monotonic deque (was O(n·w)) — ~10× faster at w=250, now flat in window size; their 2-D versions and `roll_mdd` run column-parallel (`numba` `prange`) and `roll_mdd` no longer allocates per window (~2×). Results are bit-identical to the previous implementation (verified against a naive reference).

### Fixed

### Deprecated

### Removed

## [2.1.0] - 2026-06-15

### Breaking Changes

- **All Cython removed** (E7 numba modernization). The `*_cy` kernels — features `momentums`/`metrics`/`roll_functions` and the ARMA/GARCH `econometric_models`/`estimator` — were ported to **Numba `@njit`**; the `*_cy` modules and their public `*_cy_1d/2d` / `MA_cy`/`ARMA_cy`/… symbols are gone. The build is now pure-Python (no `setup.py`, no compile step).

### Added

- `BaseNeuralNet.fit(X, y, epochs)` and array-like `predict` — every NN model (MLP/RNN/GRU/LSTM/TCN/Transformer) now conforms to the `SignalModel` protocol and composes with `fynance.strategy.Strategy`.

### Changed

- Numerical kernels run on Numba `@njit` instead of Cython (parity verified to 1e-9/1e-10 against the former Cython via golden-value tests). `pyproject.toml` build-system no longer requires Cython.

### Fixed

- `roll_annual_volatility` (and therefore `roll_sharpe`) returned `NaN` for windowed cases: the Cython kernel aliased its return buffer with the output array (`cdef double[:] R = var`), corrupting the rolling sums. The Numba port uses a separate buffer and computes correctly.

### Deprecated

### Removed

- `setup.py`, all `*_cy.pyx`/`.c` modules, and the Cython build machinery.

## [2.0.0] - 2026-06-15

### Breaking Changes

- **2.0 refactor** into a layered ML/DL backtesting tool. No compatibility shims; see [`doc/MIGRATION-2.0.md`](doc/MIGRATION-2.0.md).
- `fynance.algorithms` renamed to **`fynance.portfolio`** (allocation + sizing).
- Performance metrics moved out of `fynance.features` into **`fynance.metrics`** (`sharpe`, `sortino`, `calmar`, `diversified_ratio`, `annual_return`/`annual_volatility`, `drawdown`, `mdd`, `perf_*`, `returns_strat`, `roll_*`); the `fynance.features.metrics` aggregator is removed. `mad`/`roll_mad` moved to `fynance.features.stats`.

### Added

- **`fynance.core`** — `PriceSeries` (thin numpy-backed financial series; composition, not `ndarray` subclassing) with price↔return identities, numpy/torch bridges and `.pipe`; pipeline protocols (`DataSource`, `FeatureTransform`, `SignalModel`, `Allocator`, `CostModel`, `Metric`).
- **`fynance.data`** — `DataSource` port + `load()` dispatcher, CSV/Parquet adapters, `align`/`resample` (causal), and no-lookahead `train_test_split`/`walk_forward` splitters.
- **`fynance.backtest`** — vectorized `backtest()` engine (positions + returns/prices + cost → `BacktestResult`), `ProportionalCost`, and `BacktestResult` with `summary()`.
- **`fynance.metrics.summary`** — one-call performance panel + `METRICS` registry.
- **`fynance.plot`** — composable matplotlib figures and a one-call `tearsheet`/`tearsheet_text`.
- **`fynance.signal`** — prediction→position mappers (`sign`, `threshold`, `rank`, `vol_target_position`) and `SignalPipeline`.
- **`fynance.strategy.Strategy`** — optional orchestrator composing the pipeline, with `run` and no-lookahead `run_walk_forward`.
- Top-level API: key names re-exported on `fynance` (`PriceSeries`, `load`, `Strategy`, `tearsheet`, `sharpe`, `summary`, …). The vectorized engine is `fynance.backtest.backtest` (the `backtest` attribute on `fynance` is the subpackage).
- Optional **Streamlit playground** (`apps/playground/`, `pip install -e ".[ui]"`).
- `Notebooks/quickstart_v2.ipynb` end-to-end tour; Sphinx pages for every 2.0 subpackage; `doc/MIGRATION-2.0.md`.

### Changed

### Fixed

### Deprecated

### Removed

- Dead legacy `fynance.core.series.Series` (a `numpy.ndarray` subclass superseded by `PriceSeries`); it was never part of the public `fynance` namespace.

## [1.6.0] - 2026-06-14

### Added

- Sphinx API docs now cover every new feature: pages for TCN, Transformer, the stacking ensemble, all loss functions, training utilities, position sizing, feature engineering and market-regime detection; README updated accordingly

- `StackingEnsemble` (`fynance.models.ensemble`) — direction + magnitude base models combined by a meta-model trained on their out-of-fold predictions (leak-free stacking)

- `detect_regimes` (`fynance.features.regime`) — k-means market-regime labelling on rolling vol/return features, ordered by volatility

- `fynance.features.engineering`: `multi_resolution` (stack a feature across windows), `granger_causality` (F-test feature filter), `IncrementalMoments` (O(1) online mean/variance)

- Robust-training utilities: `purge` parameter on `_RollingBasis._fold_slices`/`cross_validate` (purged walk-forward CV), and `fynance.models.training` with `exp_sample_weights` and `EarlyStopping`

- `roll_rank` (`fynance.features.scale`) — rolling percentile-rank feature normalization (causal, outlier-robust)

- New differentiable losses in `fynance.models.loss`: `CalmarLoss` (Calmar via `torch.cummax` drawdown), `OmegaLoss` (gain/loss ratio over a threshold), and `HybridLoss` (convex combo of two losses, with an optional learnable weight)

- Realistic-backtest primitives: robustness metrics `percent_positive` / `tail_ratio` (`fynance.features`), and `fynance.algorithms.sizing` with `kelly_fraction`, causal `vol_target`, and turnover-based `transaction_cost`

- Technical indicators in `fynance.features.indicators`: `roc`, `realized_volatility`, `rolling_skewness`, `rolling_kurtosis`, `rolling_autocorr` — single-series, strictly causal, with parity + no-lookahead tests

### Changed

### Fixed

### Deprecated

### Removed

## [1.5.0] - 2026-06-14

### Added

- `Notebooks/pytorch_examples.ipynb` — a runnable PyTorch tour (metrics, allocation, MLP/TCN/Transformer with `SharpeLoss`, walk-forward CV, custom losses), replacing the old Keras/dev notebooks

- `Transformer` (`fynance.models.transformer`) — a causal Transformer encoder on `BaseNeuralNet` (sinusoidal `PositionalEncoding`, reuses `MultiHeadAttention`, lower-triangular causal mask = no lookahead), with 10 tests incl. a causality check; works with MSE and `SharpeLoss`

- `TemporalConvNet` (`fynance.models.tcn`) — a causal dilated Temporal Convolutional Network on `BaseNeuralNet` (residual blocks, dilation 1/2/4…, strictly no-lookahead), with 8 tests incl. a causality check; works with MSE and `SharpeLoss`

- Integration test training `RollMultiLayerPerceptron` with the differentiable `SharpeLoss` (instead of MSE) — demonstrates the custom financial losses end-to-end

### Changed

- `HRP()` scatters cluster-ordered weights via NumPy fancy-indexing instead of a Python loop

### Fixed

- README quickstart: `fy.ERC(cov)()` → `fy.ERC(cov)` (ERC returns an array, not a callable) and the rolling example now uses the real `model(train_period=…, test_period=…, roll_period=…)` signature

- Swept stale/decided inline `# TODO`/`# FIXME` markers; in particular the `momentums_cy` "window is w+1" FIXME was verified stale (window size is correct, proven by the property tests) and removed

### Deprecated

### Removed

- `xgboost` dependency — it was declared in `pyproject.toml` but never imported anywhere in the package (lighter install; non-breaking)

- Legacy notebooks: the Keras `Exemple_Rolling_NeuralNetwork.ipynb` and the stale dev notebooks `test_roll_NN_test.ipynb` / `Test_various_NN_models_with_simulated_data.ipynb`

## [1.4.0] - 2026-06-14

### Added

- Tests for the causal core: econometric `MA`/`ARMA` recurrences and `ARMAX_GARCH` properties; `RollMultiLayerPerceptron._training` (loss update) and `get_stats` (populated structured array)
- Property test suite `tests/features/test_property.py`: independent NumPy-reference parity for the rolling kernels (`sma`/`wma`/`smstd`/`ema`/`roll_min`/`roll_max`) and a generic no-lookahead (causality) check
- `polars` is now accepted as an input frame wherever pandas was (`BaseNeuralNet.set_data`, `econometric_models.MA`), alongside numpy/torch
- Golden-value regression test for `rolling_allocation` (previously untested)
- Release workflow now creates a **GitHub Release** on a `v*` tag (a
  `github-release` job that extracts the matching `CHANGELOG.md` section and
  publishes it via `softprops/action-gh-release`, `make_latest`), alongside the
  existing PyPI publish. (#57)

### Changed

- `mypy` is now clean (0 errors, was 103) and **enforced in CI** (new `typecheck` job). Real fixes: `print_stats` no longer reuses the `perf` flag as an array; `BaseNeuralNet.set_data` uses `X.shape[1]` (works for numpy/torch/polars, not just tensors); `perf_returns` error message fixed. `warn_return_any` disabled (numpy/Cython pervasively return `Any`); remaining structural cases (torch multiple-inheritance mixins, decorator-filled `w`, star-imported Cython names) carry targeted `# type: ignore`
- `features/metrics.py` (1782 lines) split by concern into `returns.py`, `ratios.py`, `drawdown.py`, `stats.py` + `_metrics_helpers.py`. `metrics.py` is kept as a thin re-export aggregator, so the public API (`fynance.*`, `fynance.features.metrics.*`), all import paths and the Sphinx docs are unchanged
- `models/rolling.py` slimmed: the `CVResult` dataclass moved to `models/cv_result.py` (re-exported; imports preserved) and the dead `get_perf` helper removed. The coupled `_RollingBasis`/`RollMultiLayerPerceptron` hierarchy is kept together by design
- `backtest/dynamic_plot_backtest.py` split: the orchestrator `BacktestNeuralNet` moved to a new `backtest/backtest_neural_net.py` (re-exported from `fynance.backtest`; existing imports preserved). The tightly-coupled `DynaPlot*` plot family stays together
- `estimator.estimation()` now raises `NotImplementedError` (it was an experimental, non-functional placeholder marked "NOT YET WORKING") and points to `models.econometric_models.get_parameters` (the Cython-backed authoritative path); unused `fmin`/`target_function_cy` imports removed
- **BREAKING**: pandas is replaced by polars at the input edges and by numpy at the output edges. `rolling_allocation` now returns `(numpy.ndarray, numpy.ndarray)` instead of `(pandas.Series, pandas.DataFrame)`, and `RollMultiLayerPerceptron.get_stats` returns a structured `numpy.ndarray` instead of a `pandas.DataFrame` (field access `stats["train_loss"]` is preserved; use `stats.size` instead of `.empty`). `rolling_allocation` was rewritten pandas-free in numpy with exact parity verified against the old implementation
- CI now enforces two extra gates on every PR: docstring coverage (`interrogate`, fail-under 80%) in the lint job, and a Sphinx HTML build with warnings-as-errors (`sphinx-build -W`) in a new `docs` job

### Fixed

- `tests/core/series.py` renamed to `test_series.py` so pytest collects it — `core.series` (the `Series` ndarray subclass) was previously untested (0% coverage) despite having a 6-test suite
- `estimator.estimation`/`target_function` now raise `ValueError(f"Unknown model: {model!r}")` instead of printing a typo'd message and raising a bare `ValueError`
- Removed leftover debug `print` statements from `models.rolling._RollingBasis._training`; training errors now propagate with their original traceback

### Deprecated

### Removed

- Dead code: `algorithms/browsers.py` (unused `BrowserData`, empty `RollingBasis` stub) + its `browsers_cy` Cython extension, and `algorithms/rolling.py` (`_RollingMechanism`, unused since `rolling_allocation` was rewritten pandas-free)
- `pandas` dependency (replaced by `polars` for input, `numpy` for output)
- Dead code: `models/basis.py` (`SignalModel` referencing an unset `self.y_pred`, empty `MagnitudeModel`) and the deprecated `__BacktestNeuralNet` / unused `_BacktestNeuralNet` stubs in `backtest/dynamic_plot_backtest.py`

## [1.3.4] - 2026-05-31

### Added

- Harmonise Sphinx doc with DCCD: sticky header with logo, hero on homepage, badge
  row, `installation.rst`, `changelog.rst`, `sidebar/related-projects`, favicons,
  light/dark logos, `sphinx.ext.viewcode`, toctrees with captions (#52)
- `quickstart.rst` — new Getting Started page covering metrics, indicators,
  portfolio allocation, rolling neural network, and custom loss functions (#53)

## [1.3.3] - 2026-05-11

### Added

- `sortino(X, rf, period, ...)` evaluation metric in `fynance.features.metrics`, symmetric to `sharpe`
- `directional_accuracy(y_true, y_pred)` evaluation metric — percentage of correctly predicted return signs
- `fynance.models.loss` submodule with differentiable PyTorch loss functions: `SharpeLoss`, `SortinoLoss`,
  `DirectionalAccuracyLoss`, and base class `BaseLoss`
- Tests for new metrics and loss functions; overall test and docstring coverage above 80%

### Changed

- `_compute_returns` factorised as a shared helper; `_annual_volatility` and `_annual_downside_volatility`
  use it — removes duplicated logic and double allocation in `sortino()`
- `accuracy()` simplified to a single numpy pass
- `BaseLoss.__init__` precomputes `rf / period` to avoid per-forward division
- `SharpeLoss` uses `std(correction=0)` for consistency with the numpy `sharpe` metric

### Fixed

- Reformat long import in `test_allocation.py` (ruff I001)

### Removed

- Dead commented-out `__add__` / `__iadd__` stubs from `LossSeries`
- `TODO.md` untracked from git (already declared in `.gitignore`)

## [1.3.2] - 2026-05-07

### Added

- Python 3.13 added to the CI test matrix
- Read the Docs configuration (`.readthedocs.yaml`) for versioned doc deployment (stable/latest)
- Test coverage badge via Codecov and docstring coverage badge via interrogate

### Changed

- CI now triggers on push and pull_request to both `master` and `develop`; concurrency group added to cancel redundant runs on `develop → master` PR
- README reorganised: badges split onto two lines (package / quality), content updated to reflect current subpackages (Kalman filter, LSTM, MultiHeadAttention)
- Split `fynance/models/` into one-file-per-class: `neural_network.py` →
  `_base.py` + `mlp.py`; `recurrent_neural_network.py` → `rnn.py`, `gru.py`,
  `lstm.py` + `_recurrent_base.py`. Private cell classes renamed to `_GRUCell`,
  `_LSTMCell`, `_RecurrentBase`; `_ForwardLayer` → `_OutputLayerMixin`. Public
  API (`fynance.models.*`) unchanged. (#38)
- Expose `GRUCell` and `LSTMCell` as public composable building blocks
  (mirrors PyTorch's `nn.GRUCell` / `nn.GRU` pattern). Both raise
  `NotImplementedError` on `train_on` / `predict`; use
  `GatedRecurrentUnit` / `LongShortTermMemory` for standalone training.
  `_RecurrentBase` now accepts `y=None` so cells can be constructed
  with an input dimension alone: `GRUCell(8, hidden_state_size=16)`. (#38)

## [1.3.1] - 2026-05-06

### Changed

- Set PyPI development status classifier to `Production/Stable`
- Update README to reflect stable subpackages

## [1.3.0] - 2026-05-06

### Changed

- `bollinger_band` legacy single-array notice reclassified from
  `UserWarning` to `DeprecationWarning`; the legacy return path is
  scheduled for removal in fynance 2.0.
- All other internal `warnings.warn(...)` calls now pass `category=` and
  `stacklevel=` explicitly (no semantic change — same `UserWarning`).

### Added

- Walk-forward cross-validation API on `_RollingBasis`: `cross_validate(model_factory, X, y, metric_fn=None, epochs=1)` accumulates out-of-fold predictions into `CVResult`; `_fold_slices()` exposes the windowing iterator as a reusable generator (`fynance/models/rolling.py`) (#31)
- 1.x **API stability policy** declared in `fynance/__init__.py` and
  `CONTRIBUTING.md`; public `__all__` of `fynance.models` is now
  frozen and built from explicit imports.
- Strict pytest `filterwarnings`: any `DeprecationWarning` /
  `PendingDeprecationWarning` raised from inside `fynance.*` is
  promoted to a test failure, so internal deprecations cannot ship
  silently.

## [1.2.0] - 2026-05-03

### Breaking Changes

- Removed legacy `fynance.neural_networks` Keras module — migrate to `fynance.models` (PyTorch)

### Added

- Kalman filter with RTS smoother and MLE parameter estimation (`fynance/features/filters.py`)
- `MultiHeadAttention` model with causal masking (`fynance/models/`)
- Complete rewrite of rolling walk-forward evaluation (`fynance/models/rolling.py`)
- Numba `@njit` optimization on Kalman filter and RTS smoother
- Type annotations on all public APIs (`NDArray[np.float64]`, return types)
- Type annotations on private helpers in `fynance/algorithms/allocation.py`
- GitHub Actions CI matrix (Python 3.10 / 3.11 / 3.12 / 3.13, Linux)
- `ruff` and `pre-commit` configuration

### Changed

- `HRP()`: removed pandas dependency, replaced with pure NumPy — results numerically identical
- `rolling_allocation()`: updated deprecated pandas 3.0 APIs (`ffill()` / `bfill()`)
- Build system migrated from `setup.py` to `pyproject.toml` (PEP 517/518)
- Version management migrated to `importlib.metadata`
- Replaced Travis-CI with GitHub Actions

### Fixed

- NumPy 2.x compatibility: removed deprecated aliases (`np.bool`, `np.int`, `np.float`, etc.)
- Pandas 2.0 compatibility: `fillna(method=)` → `ffill()` / `bfill()`
- Cython 3 compatibility: `**` operator in `momentums_cy.pyx` returned `complex`
- Matplotlib 3.6+ compatibility: `'seaborn'` style → `'seaborn-v0_8'`
- PyTorch: `nn.Softmax()` without `dim=` in recurrent models → `nn.Softmax(dim=-1)`
- `conftest.py`: `np.set_printoptions(legacy='1.25')` for stable doctest output across NumPy versions

[1.2.0]: https://github.com/ArthurBernard/Fynance/compare/1.0.5...v1.2.0
