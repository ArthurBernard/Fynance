---
plan: v2-refactor/E2-data
kind: leaf
status: done
complexity: medium
deps: [2]
parallel: true
---

# E2.05 — temporal splits (train/test + walk-forward)

`fynance/data/split.py` — strictly time-ordered splits for ML evaluation. No
shuffling, ever. Reuses the walk-forward semantics of `_RollingBasis` but as a
pure, data-side index generator (decoupled from any model).

## API
- `train_test_split(n, *, test_size, gap=0)` → `(train_idx, test_idx)` arrays,
  with an optional embargo `gap` between train end and test start.
- `walk_forward(n, *, train, test, step, purge=0)` → generator of
  `(train_idx, test_idx)` windows: train `[t-train:t]`, test `[t:t+test]`,
  optional `purge` removed at the boundary. Mirrors `cross_validate(..., purge=)`.

## Files
- new `fynance/data/split.py`
- new `fynance/tests/data/test_split.py`

## Test
- every test index strictly > its train indices (no leakage); gap/purge create
  the expected embargo; window count == expected for given n/train/test/step;
  property test: no index appears in both train and test of the same fold.

## Done when
- splits green; no-lookahead property holds across all generated folds.
