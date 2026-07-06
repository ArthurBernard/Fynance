---
plan: v2-refactor/E3-features-metrics
kind: leaf
status: done
complexity: low
deps: [2]
parallel: false
---

# E3.03 — causality / parity test sweep

Carry over and extend the property-test battery so the reorg preserves the two
invariants: **free-fn ↔ transform-class parity** and **no-lookahead** across all
features/metrics.

## Coverage
- property test (hypothesis or fixed grids): for every rolling feature, value at
  index t depends only on X[:t+1]; perturbing X[t+1:] leaves output[:t+1] unchanged.
- parity: every `FeatureTransform` == free function; every relocated metric ==
  golden pre-move value.

## Files
- `fynance/tests/features/test_property.py` (extend), `tests/metrics/test_*`.

## Test / Done when
- the sweep is green and explicitly covers each public feature + metric;
  CI 4 gates green; this leaf removes the E3 roadmap line.
