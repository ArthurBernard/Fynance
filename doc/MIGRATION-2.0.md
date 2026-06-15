# Migrating to fynance 2.0

fynance 2.0 is a **breaking** release that turns the toolbox into a layered,
composable backtesting tool. There are **no compatibility shims** — update your
imports per the map below.

## Import-path map

| 1.x | 2.0 |
|-----|-----|
| `fynance.algorithms` | **`fynance.portfolio`** (allocation + sizing) |
| `fynance.features.metrics` (Sharpe, Sortino, Calmar, drawdown, mdd, annual_return/volatility, perf_strat, …) | **`fynance.metrics`** |
| `fynance.features` `mad` / `roll_mad` | **`fynance.features.stats`** (dispersion stats) |
| ad-hoc plotting in `fynance.backtest` | **`fynance.plot`** (`tearsheet`, `plot_equity`, …) |

Most public names are also re-exported at the top level, so `fy.sharpe`,
`fy.ERC`, `fy.PriceSeries`, `fy.Strategy`, `fy.tearsheet`, `fy.backtest` work
directly.

## New maillons (2.0)

- **`fynance.core`** — `PriceSeries` (thin numpy-backed series; **not** a
  pandas/ndarray subclass) and the pipeline protocols.
- **`fynance.data`** — `load(path)` for CSV/Parquet, `align`/`resample`,
  `train_test_split`/`walk_forward` (no-lookahead).
- **`fynance.signal`** — prediction → position mappers.
- **`fynance.backtest.backtest`** — vectorized engine → `BacktestResult`.
- **`fynance.strategy.Strategy`** — optional end-to-end orchestrator.

## Behavioural changes

- **numpy everywhere; no pandas.** Inputs accept polars/numpy; outputs are
  numpy / `PriceSeries`. (pandas was already removed in 1.x at the edges.)
- **Causality is explicit.** The backtest engine shifts positions one step
  (the position decided at `t` earns the return at `t+1`); `PriceSeries.pnl`
  does the same. Walk-forward refits on the train slice only.
- **`fynance.backtest`** is now the engine + `BacktestResult` + cost models;
  the legacy plotting stack is superseded by `fynance.plot`.

## Example

```python
import numpy as np
import fynance as fy

prices = fy.load("prices.csv")            # was: manual numpy / pandas
strat = fy.Strategy(
    features=lambda p: np.sign(np.diff(p, prepend=p[0])),
    signal=lambda x: x,
    cost=fy.ProportionalCost(fee=0.0005),
)
result = strat.run(prices)
print(result.summary())                    # was: scattered metric calls
fig = fy.tearsheet(result)                  # was: manual plotting
```
