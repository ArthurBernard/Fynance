---
plan: v2-refactor/E6-signal-portfolio
kind: global
status: planning
roadmap: "E6 signal+portfolio — signal/ mappers and portfolio/ (ex-algorithms) under protocols"
release_on_done: false
---

# E6 — signal + portfolio

Two reorgs that complete the maillons between model and backtest. `signal/`
turns a model prediction into a position; `portfolio/` (renamed from
`algorithms/`) holds allocation + sizing under the `Allocator` protocol.

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | `signal/` mappers | medium | — |
| 02 | `portfolio/` (rename algorithms) + Allocator | medium | — |

Both depend on E1. Parallel-able.
