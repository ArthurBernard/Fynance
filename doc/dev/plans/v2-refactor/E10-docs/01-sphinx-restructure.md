---
plan: v2-refactor/E10-docs
kind: leaf
status: done
complexity: medium
deps: []
parallel: false
---

# E10.01 — Sphinx restructure for the new layout

Rebuild `doc/source/` to mirror the 2.0 package tree: `core`, `data`, `features`,
`metrics`, `models`, `signal`, `portfolio`, `backtest`, `plot`, `strategy`.

## Scope
- new subpackage `.rst` (card grid + hidden toctree) + module `.rst`
  (`currentmodule` + `autosummary :toctree: generated/`) for every new package.
- drop pages for removed modules (algorithms, backtest plotting, `*_cy`).
- regenerate committed autosummary stubs; keep `sphinx-build -W` clean.

## Files
- `doc/source/*.rst`, `doc/source/generated/*`.

## Test / Done when
- `cd doc && make html` (and `-W`) clean; every public symbol documented; CI docs
  gate green.
