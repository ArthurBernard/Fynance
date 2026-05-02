# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dev setup from scratch
pip install -e ".[dev]" && python setup.py build_ext --inplace

# Activate git hooks (run once per clone)
git config core.hooksPath .githooks

# Build Cython extensions (required after editing any .pyx file)
python setup.py build_ext --inplace

# Run full test suite (--doctest-modules --exitfirst -vv configured in pyproject.toml)
pytest

# Run a single test function
pytest fynance/tests/features/test_metrics.py::test_accuracy -v

# Run only doctests for a subpackage
pytest --doctest-modules fynance/features/

# Run with coverage
pytest --cov=fynance --cov-report=term-missing

# Lint
ruff check fynance/

# Build Sphinx docs
cd doc && make html
```

## Git Flow

**Branch model:**
```
master          ← stable releases only (tagged vX.Y.Z)
  └── develop   ← integration branch
        ├── feat/<topic>   new feature or modernization axis
        ├── fix/<topic>    bug fix
        ├── chore/<topic>  tooling, CI, deps
        └── docs/<topic>   documentation only
```

**Rules — always follow these before committing or pushing:**
1. **Never commit directly to `master`.**
2. **Never commit directly to `develop`** for non-trivial work — use a feature branch.
3. Branch off `develop` (not `master`): `git checkout develop && git checkout -b feat/my-topic`
4. Open a PR into `develop` when done.
5. `develop` → `master` only at release time.

**Commit style (Conventional Commits):**
```
feat: add TCN model
fix: replace ** in momentums_cy.pyx for Cython 3 compat
chore: migrate to pyproject.toml
docs: update CONTRIBUTING
```

**Before every commit:** run `pytest` and `ruff check fynance/`. Both must pass.

## Architecture

### Cython / Python dual-implementation (features)

`fynance/features/` has two files per computation: `metrics_cy.pyx` (Cython, compiled) and `metrics.py` (pure Python). The `__init__.py` imports both. New performance-critical code should use **Numba `@njit`** — not new Cython — and live in the Python file alongside the existing implementation.

The `USE_CYTHON='auto'` guard in `setup.py` tries to compile `.pyx` sources with Cython; if unavailable it falls back to pre-compiled `.c` files. Do not break this fallback when touching `setup.py`.

### Rolling / walk-forward pattern

`_RollingBasis` in `fynance/models/rolling.py` is the base for all walk-forward evaluation. It behaves as an iterator: `__call__` sets window parameters (`n` = train length, `s` = test length, `r` = roll step), and each `__next__` call trains on `X[t-n:t]` and predicts on `X[t:t+s]`. `RollMultiLayerPerceptron` subclasses this. `rolling_allocation()` in `fynance/algorithms/` replicates the same pattern as a function decorator for portfolio methods.

### Estimator → models pipeline

`fynance/estimator/estimator_cy.pyx` is the Cython ARMA/GARCH parameter estimator. `fynance/models/econometric_models.py` wraps it via `get_parameters()`. Do not duplicate parameter estimation logic in the Python layer.

### PyTorch models vs legacy Keras

| Location | Framework | Status |
|---|---|---|
| `fynance/models/` | PyTorch (`BaseNeuralNet`, MLP, LSTM, attention) | Active — extend here |
| `fynance/neural_networks/` | Keras / TensorFlow | Legacy — migrate to `models/`, do not extend |

When migrating from `neural_networks/`, replicate the rolling iterator interface (`__call__`/`__iter__`/`__next__`) so call-sites are unaffected.

## Modernization constraints

- **Performance**: Numba `@njit` for new numerical code. No new Cython unless wrapping a C library.
- **ML**: PyTorch only. Do not extend Keras/TensorFlow code.
- **Architecture targets**: LSTM, TCN, Transformers over rolling MLP; walk-forward CV in training loops; custom loss functions targeting Sharpe/Sortino/directional accuracy.
- **Build**: `pyproject.toml` is the authoritative build config. `setup.py` handles Cython extensions only.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`). Travis-CI removed.

## Stable vs. in-progress subpackages

| Subpackage | Policy |
|---|---|
| `fynance.features` | Extend only — never rewrite Cython code |
| `fynance.algorithms.allocation` | Stable public API — deprecation path required for breaking changes |
| `fynance.backtest` | Improve freely |
| `fynance.models` | Modernize ML architecture freely |
| `fynance.neural_networks` | Legacy — migrate then deprecate, do not add features |

## Testing conventions

Tests live in `fynance/tests/` mirroring the subpackage structure. They use `pytest` fixtures and plain `assert` statements. The suite also runs all doctest examples via `--doctest-modules`, so keep docstring examples correct and runnable.

No lookahead bias: any test involving time-series data must respect strict temporal ordering — no shuffling, no future data leaking into training windows.
