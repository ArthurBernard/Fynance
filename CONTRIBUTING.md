# Contributing to Fynance

## Setup

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/ArthurBernard/Fynance.git
cd Fynance
pip install -e ".[dev]"

# Activate the project git hooks (run once per clone)
git config core.hooksPath .githooks
```

The build is **pure-Python** — there is no compile step and no `setup.py`.
Numerical kernels are Numba `@njit` (JIT-compiled on first call), and
`pyproject.toml` is the authoritative build config.

## Git Flow

```
master          ← stable releases only (tagged vX.Y.Z, published to PyPI)
  └── develop   ← integration branch for ongoing modernization
        ├── feat/<topic>    new feature or modernization axis
        ├── fix/<topic>     bug fix
        ├── chore/<topic>   tooling, CI, deps, refactor
        └── docs/<topic>    documentation only
```

**Rules:**
- Never commit directly to `master` — always go through `develop` via a PR.
- Never commit directly to `develop` for non-trivial work — use a feature branch.
- Branch off `develop`, not `master`.
- Merge back into `develop` when the feature is complete and tests pass.
- `develop` → `master` happens only at release time (version bump + tag).

**Branch naming:** `feat/`, `fix/`, `chore/`, `docs/` + short kebab-case description.
Examples: `feat/transformer-model`, `fix/numpy2-compat`, `chore/pyproject-migration`.

**Commit style:** [Conventional Commits](https://www.conventionalcommits.org/)
```
feat: add TCN model to fynance/models/
fix: clamp adaptive window to avoid lookahead on short series
chore: migrate setup.py metadata to pyproject.toml
docs: add Git Flow section to CONTRIBUTING.md
```

## Architecture

| Subpackage | Description | Policy |
|---|---|---|
| `fynance.features` | Indicators, momentums, filters, scaling, regime, money management (Numba kernels) | Extend freely — kernels are Numba `@njit` |
| `fynance.metrics` | Risk-adjusted ratios, drawdown, returns, `summary` | Extend freely |
| `fynance.portfolio` | Portfolio allocation (HRP, MVP, ERC, IVP, MDP) + sizing | Stable public API — deprecation path required |
| `fynance.models` | PyTorch nets (MLP, RNN/GRU/LSTM, attention, TCN, Transformer) + econometric ARMA/GARCH + losses | Modernize freely |
| `fynance.backtest` | Vectorized engine, cost models, `BacktestResult` | Improve freely |
| `fynance.estimator` / `fynance.models.econometric_models` | Numba ARMA/GARCH parameter estimation | Single implementation — do not duplicate parameter logic |

See [`doc/dev/04-subpackages.md`](doc/dev/04-subpackages.md) for the full
policy matrix (including `core`, `data`, `signal`, `plot`, `strategy`,
`research`).

## Code conventions

**Numerical code:** new performance-critical functions go in the `.py` file using Numba `@njit`. There is **no Cython** in the package (since 2.1 the former `*_cy.pyx` kernels were ported to Numba); only add Cython to wrap a C library.

**ML:** PyTorch only. Do not extend the legacy Keras/TensorFlow code.

**Type annotations:** all public APIs must be annotated. Use `numpy.typing.NDArray[np.float64]` for array arguments and return types.

**Comments:** only when the *why* is non-obvious (a hidden constraint, a workaround, a subtle invariant). Do not describe what the code does.

**Docstring examples:** must be correct and runnable — the test suite runs them via `--doctest-modules`.

## Running tests

```bash
# Full suite (doctests + unit tests)
pytest

# Single file
pytest fynance/tests/features/test_metrics.py -v

# With coverage
pytest --cov=fynance --cov-report=term-missing
```

Tests must pass before opening a PR. No lookahead bias in time-series tests (no shuffling, no future data leaking into training windows).

## Linting

```bash
ruff check fynance/
```

## Stability and deprecations

> **2.0 was a breaking release** (layered architecture, import-path map in
> [`doc/MIGRATION-2.0.md`](doc/MIGRATION-2.0.md) — e.g. `fynance.algorithms` →
> `fynance.portfolio`, performance metrics → `fynance.metrics`). The notes below
> describe the stability contract **within the 2.x series**.

Per-subpackage change budgets are governed by the policy matrix in
[`doc/dev/04-subpackages.md`](doc/dev/04-subpackages.md). The headline rule:
`fynance.portfolio.allocation` (ERC/HRP/IVP/MDP/MVP) and
`fynance.estimator` / `fynance.models.econometric_models` are the **stable
public surface** — within 2.x:

- public function and class signatures there are frozen — no removals, no
  backward-incompatible signature changes;
- behavioural changes that would break user code go through **one
  release** of `DeprecationWarning` before becoming the default;
- internal helpers (names prefixed with `_`) and the freely-evolving layers
  (`features`/`metrics`/`models`/`backtest`/`plot`/`strategy`/`research`) may
  change without a deprecation cycle, additively.

Active deprecations are tracked in [CHANGELOG.md](CHANGELOG.md).

### Emitting a deprecation (contributors)

```python
import warnings

warnings.warn(
    "old_function() is deprecated and will be removed in fynance 3.0; "
    "use new_function() instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
```

The CI configuration (`pyproject.toml::tool.pytest.ini_options.filterwarnings`)
promotes any `DeprecationWarning` raised from inside `fynance.*` to a
test failure, so a new deprecation cannot land without the matching
test being explicitly marked:

```python
import pytest

@pytest.mark.filterwarnings(
    "ignore:old_function:DeprecationWarning"
)
def test_old_function_still_works():
    ...
```

### Silencing fynance deprecations (downstream users)

If you depend on a deprecated path and need to stay on 2.x while you
migrate, opt out **explicitly**:

```python
import warnings
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"fynance\..*"
)
```

To go the other way and have your own test suite fail on any fynance
deprecation:

```bash
pytest -W "error::DeprecationWarning:fynance\\..*"
```

## Release process (maintainer only)

1. All planned features merged into `develop`, CI green.
2. Bump version in `pyproject.toml`.
3. Update `CHANGELOG.md`.
4. Open PR `develop` → `master`.
5. After merge: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. Publish to PyPI: `python -m build && twine upload dist/*`.
