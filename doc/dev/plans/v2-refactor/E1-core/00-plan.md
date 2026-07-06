---
plan: v2-refactor/E1-core
kind: global
status: done
roadmap: "E1 core — PriceSeries value object, Protocol seams, numpy/torch bridges"
release_on_done: false
---

# E1 — core: the spine

Foundation for everything. Introduces `fynance/core/` with the central value
object (`PriceSeries`), the `typing.Protocol` seams the whole library composes
through, and the numpy/torch bridges.

**Design invariants (apply to every leaf):**
- numpy-backed, immutable-ish value object; **compose, never subclass `ndarray`**.
- pytorch never stored — only produced on demand via `.to_torch()`.
- Thin object: ~5 core methods + `.pipe`; richness lives in free functions.

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | `PriceSeries` container (values+index+meta) | medium | — |
| 02 | finance core methods (returns/prices/pnl) | medium | 01 |
| 03 | array bridges + `.pipe()` accessor | low | 01 |
| 04 | Protocol seams module | medium | — |

Order: 01 → {02, 03}; 04 independent (can run first/parallel).
