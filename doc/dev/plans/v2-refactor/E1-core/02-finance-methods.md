---
plan: v2-refactor/E1-core
kind: leaf
status: done
complexity: medium
deps: [1]
parallel: false
---

# E1.02 — finance core methods on `PriceSeries`

Add the **few fundamental** price↔return identities as methods. Everything else
stays a free function reachable via `.pipe` (E1.03). Resist scope creep here.

## Methods (numpy under the hood, return `PriceSeries` unless noted)
- `to_returns(kind="pct" | "log" | "raw")` — pct: `p_t/p_{t-1}-1`; log:
  `ln(p_t/p_{t-1})`; raw: `p_t-p_{t-1}`. First element NaN or dropped (param
  `dropna=True`). **Causal**: no future leak.
- `to_prices(base=1.0, kind="pct"|"log")` — inverse: cumulate returns → price path.
- `cumulative()` — `(1+r).cumprod()` style equity from a return series.
- `pnl(positions)` — element-wise `position_{t-1} * return_t` (positions **shifted**
  one step → strictly causal); returns a `PriceSeries` of strategy returns.
- `drop_na()` / `fillna(method)` minimal helpers.

## Files
- extend `fynance/core/price_series.py`
- extend `fynance/tests/core/test_price_series.py`

## Test
- `to_returns(log)` then `to_prices(log)` round-trips to original (atol 1e-12);
  pct/log/raw numerically correct on a tiny hand series; `pnl` uses shifted
  positions (assert no-lookahead: a position known only at t cannot affect r_t);
  doctests on each method.

## Done when
- methods green incl. round-trip + causality assertions; mypy clean.
