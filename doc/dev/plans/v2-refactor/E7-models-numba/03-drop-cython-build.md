---
plan: v2-refactor/E7-models-numba
kind: leaf
status: todo
complexity: medium
deps: [1, 2]
parallel: false
---

# E7.03 — drop Cython from the build

With no `.pyx` left, simplify the build to pure-python + numba.

## Scope
- remove the `USE_CYTHON`/ext-modules machinery from `setup.py` (or delete
  `setup.py` if `pyproject.toml` can fully own the now-pure-python build).
- drop Cython from build deps in `pyproject.toml`; add `numba` to runtime deps.
- update CI: no `build_ext` step; update CLAUDE.md "Commands" (no Cython build).
- update `.github/workflows` and any `build_ext --inplace` references.

## Files
- `setup.py`, `pyproject.toml`, `.github/workflows/*.yml`, `CLAUDE.md`, docs.

## Test
- clean `pip install -e ".[dev]"` then `pytest` green with **no** compile step;
  fresh clone import works without a C toolchain.

## Done when
- build is pure-python+numba; CI green; docs/commands updated.
