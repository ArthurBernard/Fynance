# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Claude-oriented developer brief**: [`doc/dev/`](doc/dev/) is an orientation
> pack written for Claude Code — overview, architecture, design decisions &
> rationale, the per-subpackage matrix, testing methodology, and current status.
> Start at [`doc/dev/README.md`](doc/dev/README.md). `CLAUDE.md` stays
> authoritative for commands and invariants.

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

**Model per task** (advisory — set via `/model` or a plan leaf's `complexity`:
`low→haiku`, `medium→sonnet`, `high→opus`):

| Model | For |
|-------|-----|
| `opus` | judgement, design, decisions, planning, review |
| `sonnet` | implementation — code, tests, docstrings |
| `haiku` | mechanical fan-out (doc scans, checklists) |

## Architecture

### Cython / Python dual-implementation (features)

`fynance/features/` has two files per computation: `metrics_cy.pyx` (Cython, compiled) and `metrics.py` (pure Python). The `__init__.py` imports both. New performance-critical code should use **Numba `@njit`** — not new Cython — and live in the Python file alongside the existing implementation.

The `USE_CYTHON='auto'` guard in `setup.py` tries to compile `.pyx` sources with Cython; if unavailable it falls back to pre-compiled `.c` files. Do not break this fallback when touching `setup.py`.

### Rolling / walk-forward pattern

`_RollingBasis` in `fynance/models/rolling.py` is the base for all walk-forward evaluation. It behaves as an iterator: `__call__` sets window parameters (`n` = train length, `s` = test length, `r` = roll step), and each `__next__` call trains on `X[t-n:t]` and predicts on `X[t:t+s]`. `RollMultiLayerPerceptron` subclasses this. `rolling_allocation()` in `fynance/algorithms/` replicates the same pattern as a function decorator for portfolio methods.

### Estimator → models pipeline

`fynance/estimator/estimator_cy.pyx` is the Cython ARMA/GARCH parameter estimator. `fynance/models/econometric_models.py` wraps it via `get_parameters()`. Do not duplicate parameter estimation logic in the Python layer.

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
| `fynance.estimator` | Do not duplicate logic in Python — Cython is authoritative |
| `fynance.backtest` | Improve freely |
| `fynance.models` | Modernize ML architecture freely |

## Testing conventions

Tests live in `fynance/tests/` mirroring the subpackage structure. They use `pytest` fixtures and plain `assert` statements. The suite also runs all doctest examples via `--doctest-modules`, so keep docstring examples correct and runnable.

No lookahead bias: any test involving time-series data must respect strict temporal ordering — no shuffling, no future data leaking into training windows.
