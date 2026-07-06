---
plan: v2-refactor/E5-reporting
kind: leaf
status: done
complexity: medium
deps: [1]
parallel: false
---

# E5.02 — `plot/` package

Move plotting out of `backtest/` into `fynance/plot/`; modernize into small,
composable figure functions (matplotlib; return `Figure`/`Axes`, never `plt.show`).

## Scope
- migrate/rewrite the useful bits of the old `backtest/plot*.py` →
  `plot/equity.py` (equity + drawdown), `plot/returns.py` (hist/QQ/rolling),
  `plot/metrics.py` (rolling sharpe/vol). Drop dead `_basis_plot`/dynamic plot
  unless trivially salvageable.
- each fn: `fn(result_or_series, *, ax=None) -> Axes`.

## Files
- new `fynance/plot/*`; delete migrated `backtest/plot*`, `print_stats` (→ metrics
  `summary` already covers stats text).
- `fynance/tests/plot/test_smoke.py` (Agg backend, no display).

## Test
- smoke: each fn returns a matplotlib `Axes`/`Figure` on a tiny `BacktestResult`
  under the Agg backend; no `plt.show`.

## Done when
- `plot/` green headless; `backtest/` no longer owns plotting; Sphinx `-W` green.
