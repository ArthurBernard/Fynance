---
plan: v2-refactor/E4-backtest
kind: global
status: done
roadmap: "E4 backtest — vectorized engine: positions + prices + costs → BacktestResult"
release_on_done: false
---

# E4 — backtest: the missing engine

Build the real artery: a **vectorized** engine that turns positions/signal +
prices + a cost model into a `BacktestResult` (equity, returns, turnover,
exposure). Strictly causal (positions shifted). Plotting is **out** (E5).

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | CostModel + proportional cost | low | — |
| 02 | vectorized engine | high | 01 |
| 03 | `BacktestResult` object | medium | 02 |

Order: 01 → 02 → 03. Depends on E1; consumes E3 metrics in 03.
