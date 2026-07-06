---
plan: v2-refactor/E10-docs
kind: leaf
status: done
complexity: medium
deps: [1, 2]
parallel: false
---

# E10.03 — README 2.0 + migration guide + CHANGELOG

Finalize user-facing docs for the breaking release.

## Scope
- rewrite `README.md` around the new pipeline + quickstart snippet (load → signal →
  backtest → tearsheet); update the subpackage table to the 2.0 layout.
- new `doc/MIGRATION-2.0.md`: import-path map (`algorithms`→`portfolio`,
  features-metrics→`metrics`, removed `*_cy`, dropped Cython build), behavioural
  breaks, the "no compat shims" note.
- `CHANGELOG.md`: a `### Breaking Changes` section enumerating the moves; this
  drives the **major** bump in `/release`.

## Files
- `README.md`, `doc/MIGRATION-2.0.md`, `CHANGELOG.md`.

## Test / Done when
- README snippet runs; migration map complete; CHANGELOG breaking section ready;
  this leaf removes the v2-refactor roadmap line → `/release` cuts **v2.0.0**.
