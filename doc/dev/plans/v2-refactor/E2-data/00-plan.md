---
plan: v2-refactor/E2-data
kind: global
status: done
roadmap: "E2 data — DataSource port, CSV/Parquet adapters, alignment, temporal splits"
release_on_done: false
---

# E2 — data: the missing ingestion maillon

The biggest gap vs the vision. A ports&adapters layer that turns local files into
`PriceSeries`, aligns/resamples them, and produces **no-lookahead** temporal
splits for ML evaluation. Polars is the read engine; numpy/`PriceSeries` the output.

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | DataSource base + registry | low | — |
| 02 | CSV adapter | medium | 01 |
| 03 | Parquet adapter | low | 01,02 |
| 04 | align / resample | medium | 02 |
| 05 | temporal splits (train/test + walk-forward) | medium | 02 |

Order: 01 → 02 → {03, 04, 05}. Depends on E1 (PriceSeries).
