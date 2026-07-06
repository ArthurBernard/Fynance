---
plan: v2-refactor/E7-models-numba
kind: leaf
status: done
complexity: high
deps: []
parallel: false
---

# E7.01 — estimator ARMA/GARCH → numba

Reimplement `fynance/estimator/estimator_cy.pyx` (ARMA/GARCH parameter estimation)
as pure-python + numba `@njit` in `fynance/estimator/estimator.py`. Authoritative
logic stays in one place (CLAUDE.md invariant), now numba not Cython.

## Approach
- port the recursion/likelihood loops to `@njit` functions; keep `get_parameters`
  public API and the `econometric_models.get_parameters()` wrapper signature.
- delete `estimator_cy.pyx` / `.c`; remove from setup.py ext list (full removal in 03).

## Files
- `fynance/estimator/estimator.py` (rewrite), delete `estimator_cy.*`,
  `models/econometric_models_cy.*` (the wrapper) — update `models/__init__`.
- `fynance/tests/estimator/test_estimator.py` (extend with golden params).

## Test
- estimated params match the Cython output on a fixed seeded series within tight
  atol (golden values captured before deletion); ARMA/GARCH/ARMA_GARCH/ARMAX_GARCH
  all covered; first-call JIT warm-up acceptable.

## Done when
- estimator numba-backed, parity-green; no estimator Cython remains.
