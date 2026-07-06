---
plan: v2-refactor/E7-models-numba
kind: leaf
status: done
complexity: low
deps: []
parallel: true
---

# E7.04 — SignalModel conformance for models

Make the pytorch models present a uniform `SignalModel` face (E1.04) so `signal/`
and `strategy/` compose them without special-casing.

## Scope
- add/confirm `fit(X, y) -> self` (wrapping the train loop) and `predict(X) ->
  ndarray` (numpy out) on `BaseNeuralNet` (or a thin adapter) so MLP/RNN/GRU/LSTM/
  TCN/Transformer/ensemble all conform.
- keep pytorch internal: numpy in, numpy out at the boundary.

## Files
- `fynance/models/_base.py` (+ adapter if cleaner); `tests/models/test_protocol.py`.

## Test
- each model isinstance `SignalModel` (runtime_checkable); `predict` returns numpy;
  a tiny fit/predict round-trip per architecture.

## Done when
- all models conform; numpy boundary holds; mypy clean.
