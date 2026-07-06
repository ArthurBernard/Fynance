---
plan: v2-refactor/E3-features-metrics
kind: global
status: done
roadmap: "E3 features+metrics — regroup features, extract performance metrics into a metrics/ package"
release_on_done: false
---

# E3 — features reorg + metrics extraction

Fix the brouillage conceptuel: **performance metrics leave `features/`** for a new
top-level `metrics/` package (evaluation ≠ feature). Features are regrouped into
clear modules and adopt the `FeatureTransform` protocol (fit/transform wrappers
over the existing free functions — functions stay composable).

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | extract perf-metrics → `fynance/metrics/` | medium | — |
| 02 | regroup features + `FeatureTransform` adoption | medium | 01 |
| 03 | causality / parity test sweep | low | 02 |

Order: 01 → 02 → 03. Depends on E1 (PriceSeries optional input).
