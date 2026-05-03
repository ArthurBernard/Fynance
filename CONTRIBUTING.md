# Contributing to Fynance

## Setup

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/ArthurBernard/Fynance.git
cd Fynance
pip install -e ".[dev]"

# Compile Cython extensions
python setup.py build_ext --inplace

# Activate the project git hooks (run once per clone)
git config core.hooksPath .githooks
```

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
Examples: `feat/transformer-model`, `fix/cython3-compat`, `chore/pyproject-migration`.

**Commit style:** [Conventional Commits](https://www.conventionalcommits.org/)
```
feat: add TCN model to fynance/models/
fix: replace ** operator in momentums_cy.pyx for Cython 3 compat
chore: migrate setup.py metadata to pyproject.toml
docs: add Git Flow section to CONTRIBUTING.md
```

## Architecture

| Subpackage | Description | Policy |
|---|---|---|
| `fynance.features` | Metrics, indicators, filters (Cython + Python) | Extend only — never rewrite Cython |
| `fynance.algorithms` | Portfolio allocation (HRP, MVP, ERC, IVP, MDP) | Stable public API — deprecation path required |
| `fynance.models` | PyTorch neural networks (MLP, GRU, LSTM, Transformer) | Modernize freely |
| `fynance.backtest` | Performance evaluation, loss series | Improve freely |
| `fynance.estimator` | Cython ARMA/GARCH parameter estimation | Do not duplicate in Python layer |

## Code conventions

**Numerical code:** new performance-critical functions go in the `.py` file using Numba `@njit`. Do not add new Cython files — Cython is only for wrapping C libraries.

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

## Cython extensions

When editing a `.pyx` file, recompile before running tests:

```bash
python setup.py build_ext --inplace
```

Do not add new Cython files. New numerical code goes in the `.py` counterpart using Numba `@njit`.

## Linting

```bash
ruff check fynance/
```

## Release process (maintainer only)

1. All planned features merged into `develop`, CI green.
2. Bump version in `pyproject.toml`.
3. Update `CHANGELOG.md`.
4. Open PR `develop` → `master`.
5. After merge: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. Publish to PyPI: `python -m build && twine upload dist/*`.
