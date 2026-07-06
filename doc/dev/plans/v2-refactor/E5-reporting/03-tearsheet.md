---
plan: v2-refactor/E5-reporting
kind: leaf
status: done
complexity: medium
deps: [1, 2]
parallel: false
---

# E5.03 — `tearsheet()` one-call report

`fynance/plot/tearsheet.py` — the single entry point that turns a `BacktestResult`
(or equity `PriceSeries`) into a full report: grid of figures + a metrics table.
This is the API the notebook and the Streamlit UI both call.

## API
- `tearsheet(result, *, metrics="standard", figsize=...) -> Figure` — composes
  equity+drawdown, rolling sharpe, returns dist, and renders `summary()` as a
  table subplot. Pure matplotlib (works headless, embeds in Streamlit).
- optional `tearsheet_text(result) -> str` for notebook/CLI quick print.

## Files
- new `fynance/plot/tearsheet.py`; export from `fynance/plot/__init__.py`
- `fynance/tests/plot/test_tearsheet.py`

## Test
- returns a `Figure` with the expected number of axes on a tiny result; the
  embedded table values equal `metrics.summary(result.returns)`.

## Done when
- `from fynance.plot import tearsheet` green headless; example used in E10 notebook.
