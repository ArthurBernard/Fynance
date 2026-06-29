# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Claude-oriented developer brief**: [`doc/dev/`](doc/dev/) is an orientation
> pack written for Claude Code — overview, architecture, design decisions &
> rationale, the per-subpackage matrix, testing methodology, and current status.
> Start at [`doc/dev/README.md`](doc/dev/README.md). `CLAUDE.md` stays
> authoritative for commands and invariants.

## Commands

```bash
# Dev setup from scratch (pure-Python build — no compile step)
pip install -e ".[dev]"

# Activate git hooks (run once per clone)
git config core.hooksPath .githooks

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
2. **Never commit directly to `develop`** — always use a feature branch + PR, even for small changes.
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

Do not add `Co-Authored-By` trailers to commits — this is a personal repo.

**Before every commit:** run `pytest` and `ruff check fynance/`. Both must pass.

**One PR = one concern, small and disposable.** Even a large plan ships as
*several* small atomic PRs — never one fourre-tout branch.

### Dev loop & docs of record

The iterative loop is tooled by user-level skills, with four tracked docs as the
sources of truth:

| Doc | Holds | Updated by |
|-----|-------|-----------|
| `doc/dev/07-roadmap.md` | open work — single source *index* (**tracked**, mirrors dccd) | `/pick-task` reads · `/finish-task`, `/abandon-task` update |
| `doc/dev/plans/<epic>/` | open work *detail* — durable plan trees (**local/gitignored**; the format `README.md` is tracked) | `/plan` writes · `/execute-leaf` reads · `/finish-task` archives |
| `doc/dev/03-decisions.md` | the *why* — ADR journal (+ settled rationale) | `/finish-task` (accepted), `/abandon-task` (rejected/tombstone) |
| `doc/dev/06-status.md` | where things stand | `/finish-task`, `/groom-docs` |

`CHANGELOG.md` + git log stay authoritative for *what* shipped. The loop:

`/pick-task` (smallest coherent slice) → `/plan` (decompose into a
`doc/dev/plans/<epic>/` tree — single leaf for a trivial task, a global
`00-plan.md` + leaves otherwise) → `/execute-leaf <epic> next` (implement +
test + verify) → `/finish-task` (tests, ADR, CHANGELOG, PR, archive the leaf) →
… per leaf … → last leaf removes the roadmap line → `/release`.
`/abandon-task` salvages the lesson + closes a bad PR; `/groom-docs` keeps
`doc/dev/` lean and true.

**Model: `opus`, always.** Per the maintainer's standing preference, **all work on
this repo runs on `opus`** — interactive sessions *and* every spawned subagent
(including `/execute-leaf`), regardless of a leaf's `complexity`. The `complexity`
tag still records effort/risk and orders the execution queue, but it **does not
downgrade the model**: treat `low | medium | high` all as `opus`.

## Architecture

### Numerical kernels — Numba (no Cython)

As of 2.1 there is **no Cython** in the package: the former `*_cy.pyx` kernels
(features `momentums`/`metrics`/`roll_functions`, and the ARMA/GARCH
`estimator`/`econometric_models`) were ported to **Numba `@njit`** living in the
`.py` modules (private `_kernel`-style functions). New performance-critical code
uses Numba too. The build is pure-Python (no compile step, no `setup.py`).

### Rolling / walk-forward pattern

`_RollingBasis` in `fynance/models/rolling.py` is the base for all walk-forward evaluation. It behaves as an iterator: `__call__` sets window parameters (`n` = train length, `s` = test length, `r` = roll step), and each `__next__` call trains on `X[t-n:t]` and predicts on `X[t:t+s]`. `RollMultiLayerPerceptron` subclasses this. `rolling_allocation()` in `fynance/portfolio/` replicates the same pattern as a function decorator for portfolio methods.

### Estimator → models pipeline

`fynance/models/econometric_models.py` holds the Numba ARMA/GARCH kernels
(`_ma`/`_arma`/`_arma_garch`/`_armax_garch`) wrapped by `MA`/`ARMA`/… and
`get_parameters()`. `fynance/estimator/estimator.py` builds on them. Keep a single
implementation — do not duplicate parameter logic.

## Modernization constraints

- **Performance**: Numba `@njit` for new numerical code. No new Cython unless wrapping a C library.
- **ML**: PyTorch only. Do not extend Keras/TensorFlow code.
- **Architecture targets**: LSTM, TCN, Transformers over rolling MLP; walk-forward CV in training loops; custom loss functions targeting Sharpe/Sortino/directional accuracy.
- **Build**: `pyproject.toml` is the authoritative (pure-Python) build config; there is no `setup.py` and no compile step.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`). Travis-CI removed.

## Stable vs. in-progress subpackages

| Subpackage | Policy |
|---|---|
| `fynance.features` | Extend freely; numerical kernels are Numba `@njit` |
| `fynance.portfolio.allocation` | Stable public API — deprecation path required for breaking changes |
| `fynance.estimator` / `fynance.models.econometric_models` | Single Numba implementation — do not duplicate parameter logic |
| `fynance.backtest` | Improve freely |
| `fynance.models` | Modernize ML architecture freely |

## Testing conventions

Tests live in `fynance/tests/` mirroring the subpackage structure. They use `pytest` fixtures and plain `assert` statements. The suite also runs all doctest examples via `--doctest-modules`, so keep docstring examples correct and runnable.

No lookahead bias: any test involving time-series data must respect strict temporal ordering — no shuffling, no future data leaking into training windows.
