# 5 — Testing

## Layers

| Layer | Command | Catches |
|-------|---------|---------|
| **Unit** | `pytest` | logic, shapes, regressions (~300 tests under `fynance/tests/`) |
| **Doctests** | `pytest --doctest-modules` (on by default in `pyproject.toml`) | every docstring `>>>` example — they are part of the suite |
| **Benchmarks** | `pytest-benchmark` (e.g. `test_filters_benchmark.py`) | perf regressions on hot paths |
| **Coverage** | `pytest --cov=fynance --cov-report=term-missing` | untested branches |
| **Property** | `pytest tests/features/test_property.py` | kernel parity vs NumPy refs + no-lookahead |
| **Type** | `mypy` | type drift (0 errors, enforced) |
| **Lint** | `ruff check fynance/` | style, dead imports |
| **Docs** | `cd doc && make html` | Sphinx (furo) builds |

`addopts` in `pyproject.toml` is `--doctest-modules --exitfirst -vv
--capture=no --ignore=fynance/dev`, so **a single failure (incl. a doctest)
aborts the run**. `conftest.py` sets NumPy legacy scalar repr
(`np.set_printoptions(legacy='1.25')`) so NumPy-2 scalar reprs don't break
existing doctests.

> **Pure-Python build.** No compile step — the package is Numba-backed (no
> Cython). A plain `pip install -e ".[dev]"` is enough before running the suite.

## Conventions (do not regress)

- **No lookahead bias.** Any test on time-series data must respect strict temporal
  ordering — no shuffling, no future data leaking into a training window. This is
  the library's core correctness property; a feature that passes its unit test but
  peeks at the future is still a bug.
- **Doctests are runnable, not decorative.** Keep every `>>>` example correct and
  deterministic (seed RNGs; mind NumPy-2 reprs).
- **Tests mirror the package.** `fynance/tests/<subpkg>/test_*.py`. Use fixtures +
  plain `assert`.
- **New features get a causal audit.** For any new rolling indicator, verify the
  computation is strictly past (`f(data[t-window:t])`) — not
  `rolling(center=True)`, not a global/whole-dataset normalisation.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs **four gates** on every PR:
the test suite across the Python matrix (`test`), `ruff` + `interrogate`
(docstring coverage ≥ 80%) in `lint`, a Sphinx build with warnings-as-errors
(`docs`, `sphinx-build -W`), and `mypy` (`typecheck`, 0 errors). `release.yml`
builds `cibuildwheel` manylinux wheels + sdist, publishes to PyPI, and creates
the GitHub Release on a `v*` tag. Badges via `badges.yml`.
