# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking Changes

### Added

### Changed

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
