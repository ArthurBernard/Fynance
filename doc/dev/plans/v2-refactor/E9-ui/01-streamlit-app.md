---
plan: v2-refactor/E9-ui
kind: leaf
status: done
complexity: medium
deps: []
parallel: false
---

# E9.01 — Streamlit playground

`apps/playground/app.py` — two-column layout: left an editor (`streamlit-ace`) for
a `signal(df) -> positions` function + a data picker; right the live `tearsheet()`.

## Scope
- left: file/data picker (loads via `fynance.data.load`), code editor seeded with a
  template signal fn; "Run" button.
- right: run the user's fn → `fynance.backtest.backtest` → `fynance.plot.tearsheet`
  rendered via `st.pyplot`; metrics table from `metrics.summary`.
- packaging: `[project.optional-dependencies] ui = ["streamlit", "streamlit-ace"]`;
  a `fynance-playground` console-script or `streamlit run apps/playground/app.py`.
- **Security note**: executes user code via `exec` — documented as local-only;
  not a hosted/multi-user surface (no sandbox in 2.0).

## Files
- new `apps/playground/app.py`, `apps/playground/README.md`; `pyproject.toml` extra.

## Test
- a smoke test importing the app module's pure helpers (signal-runner) with a
  template fn → produces a `BacktestResult` (no Streamlit runtime needed); the
  Streamlit layer itself is manual/CI-light.

## Done when
- `streamlit run apps/playground/app.py` shows code→tearsheet locally; extra
  documented; helper smoke test green.
