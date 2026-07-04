---
plan: v2-refactor/E5-reporting
kind: global
status: done
roadmap: "E5 reporting — metrics consolidation, plot/ package, one-call tearsheet()"
release_on_done: false
---

# E5 — reporting: metrics + plot + tearsheet

The maillon that makes both the notebook workflow and the future UI trivial. A
clean `tearsheet(result)` is the real product value; the Streamlit UI (E9) is a
thin shell over it.

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | metrics consolidation + `Metric` protocol | medium | — |
| 02 | `plot/` package (figures) | medium | 01 |
| 03 | `tearsheet()` one-call report | medium | 01,02 |

Depends on E3 (metrics moved) and E4 (`BacktestResult`).
