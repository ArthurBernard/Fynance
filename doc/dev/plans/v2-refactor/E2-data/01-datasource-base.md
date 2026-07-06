---
plan: v2-refactor/E2-data
kind: leaf
status: done
complexity: low
deps: []
parallel: false
---

# E2.01 — DataSource base + registry

Concretize the `DataSource` port (E1.04) with a small base + a name→adapter
registry so callers do `load("data.csv")` without importing adapters directly.

## API
- `BaseDataSource` (ABC implementing the `DataSource` Protocol) with `load(...)`.
- `register(name)` decorator + `get_source(name)` / `load(path, source="auto", **kw)`
  dispatcher that auto-selects by file extension (`.csv`→csv, `.parquet`→parquet).

## Files
- new `fynance/data/__init__.py`, `fynance/data/base.py`
- new `fynance/tests/data/test_registry.py`

## Test
- registry dispatch by extension; unknown extension → clear `ValueError`.

## Done when
- `from fynance.data import load` works; registry tests green.
