# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Deprecated

### Removed

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
