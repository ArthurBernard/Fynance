---
plan: v2-refactor/E8-strategy
kind: global
status: done
roadmap: "E8 strategy — optional orchestrator composing data→features→signal→portfolio→backtest→metrics"
release_on_done: false
---

# E8 — strategy: the optional orchestrator

Compose the protocol-based maillons into one runnable `Strategy` — **optional**,
never required (the maillons stay usable standalone). Provides the end-to-end and
walk-forward runs the UI and the example notebook drive.

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | `Strategy` object (compose protocols) | high | — |
| 02 | walk-forward run | medium | 01 |

Depends on E2,E3,E4,E6,E7. Decision **D4** (API style) decided in leaf 01.
