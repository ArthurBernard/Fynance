---
plan: v2-refactor/E5-reporting
kind: leaf
status: done
complexity: medium
deps: []
parallel: false
---

# E5.01 — metrics consolidation + `Metric` protocol

Finish the `fynance/metrics/` package started in E3: ensure a complete, coherent,
numpy-only metric set, all conforming to the `Metric` protocol, with a registry.

## Scope
- complete set: sharpe, sortino, calmar, omega, mdd, max_drawdown_duration,
  annual_return, annual_volatility, downside_vol, var/cvar, hit_rate,
  percent_positive, tail_ratio, turnover, rolling variants (`roll_sharpe`).
- a `METRICS` registry + `summary(returns) -> dict` computing the standard panel.
- consistent signature `metric(returns, *, period=252, **kw) -> float`.

## Files
- `fynance/metrics/*` (extend), `fynance/metrics/registry.py`
- `fynance/tests/metrics/test_summary.py`

## Test
- golden values on a fixed return series; registry round-trip; `summary` keys
  stable; each metric isinstance `Metric` (runtime_checkable).

## Done when
- metrics package complete + green; mypy clean.
