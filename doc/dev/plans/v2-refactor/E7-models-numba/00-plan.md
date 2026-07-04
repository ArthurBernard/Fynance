---
plan: v2-refactor/E7-models-numba
kind: global
status: planning
roadmap: "E7 models+numba — estimator Cython→numba, drop Cython build, SignalModel conformance"
release_on_done: false
---

# E7 — models + numba migration

Modernize the numerical backend per the 2.0 decision (drop Cython → numba) and
make the model layer conform to the `SignalModel` protocol. pytorch stays the ML
backend, confined to this maillon.

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | estimator ARMA/GARCH → numba | high | — |
| 02 | features `*_cy` → numba (D1) | high | 01 |
| 03 | drop Cython from build | medium | 01,02 |
| 04 | SignalModel conformance for models | low | — |

Depends on E1. 02 is gated on decision **D1** (confirm migrate vs freeze).
