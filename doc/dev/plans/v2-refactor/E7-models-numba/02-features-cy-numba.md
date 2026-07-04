---
plan: v2-refactor/E7-models-numba
kind: leaf
status: done
complexity: high
deps: [1]
parallel: false
---

# E7.02 — features `*_cy` → numba  (decision D1)

**Gate: confirm D1** (2.0 drops the Cython/Python duality in `features/`). If
confirmed, migrate the hot loops of `momentums_cy`/`metrics_cy` etc. to numba and
delete the `.pyx`/`.c` + the dual import in `features/__init__.py`.

## Approach
- for each `*_cy` function, ensure the pure-python `.py` twin is numba-accelerated
  (`@njit` on the inner loop), benchmark vs Cython (must be within ~1.5×), then
  drop the `_cy` variant and its `__init__` import.
- keep the public function names identical (single implementation now).

## Files
- `fynance/features/{momentums,metrics,...}.py` (numba), delete `*_cy.pyx/.c`,
  edit `features/__init__.py`.
- bench note in the PR; tests already cover correctness (E3.03 sweep).

## Test
- parity with previous outputs (golden); a micro-benchmark recorded; suite green.

## Done when
- no `features/*_cy` remains; numba twins are the single source; perf acceptable.
