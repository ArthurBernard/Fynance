---
plan: v2-refactor/E1-core
kind: leaf
status: done
complexity: low
deps: [1]
parallel: true
---

# E1.03 — array bridges + `.pipe()` accessor

Make `PriceSeries` interoperate cleanly and reach the free-function toolbox
without becoming a god-object.

## API
- `.to_numpy(copy=True) -> np.ndarray`
- `.to_torch(dtype=torch.float32, device=None) -> torch.Tensor` (lazy torch import)
- `.pipe(fn, *args, **kwargs)` — calls `fn(self.values, *args, **kwargs)`; if the
  result is array-like of same length, wrap back into a `PriceSeries` (same index),
  else return raw. This is the delegation pattern: `ps.pipe(rsi, window=14)`.
- optional `.apply(fn)` element-wise convenience.

## Files
- extend `fynance/core/price_series.py`
- extend `fynance/tests/core/test_price_series.py`

## Test
- `to_torch` dtype/shape; torch import is lazy (no hard import at module load);
  `.pipe` wraps length-preserving results, passes through scalars; index carried.

## Done when
- bridges + pipe green; importing `fynance.core` does not import torch eagerly.
