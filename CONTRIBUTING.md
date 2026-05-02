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

## Running tests

```bash
# Full suite (doctests + unit tests)
pytest

# Single file
pytest fynance/tests/features/test_metrics.py -v

# With coverage
pytest --cov=fynance --cov-report=term-missing
```

Tests must pass before opening a PR. No lookahead bias in time-series tests.

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
2. Bump version in `pyproject.toml` and `fynance/version.py`.
3. Update `CHANGELOG.md`.
4. Open PR `develop` → `master`.
5. After merge: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. Publish to PyPI: `python -m build && twine upload dist/*`.
