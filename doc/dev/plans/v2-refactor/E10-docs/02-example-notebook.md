---
plan: v2-refactor/E10-docs
kind: leaf
status: done
complexity: low
deps: []
parallel: true
---

# E10.02 — end-to-end example notebook

`Notebooks/quickstart_v2.ipynb` — the reference workflow: load CSV → build features
→ train a model (or write a signal fn) → backtest → `tearsheet`. Doubles as the
"notebook does the job" answer and the template the Streamlit app seeds.

## Files
- new `Notebooks/quickstart_v2.ipynb`; remove stale 1.x notebooks.

## Test / Done when
- every cell runs top-to-bottom (verify by executing the concatenated cell code);
  produces a tearsheet figure; uses only public 2.0 API.
