---
plan: v2-refactor/E10-docs
kind: global
status: done
roadmap: "E10 docs — Sphinx restructure, end-to-end notebook, README 2.0 + migration guide"
release_on_done: true
---

# E10 — docs, examples, migration (closes 2.0)

Bring all documentation in line with the new layout and tell users how to move
from 1.x. Cross-cutting; lands near the end. Last leaf removes the v2-refactor
roadmap line and triggers `/release` → v2.0.0.

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | Sphinx restructure for new layout | medium | — |
| 02 | end-to-end example notebook | low | — |
| 03 | README 2.0 + migration guide + CHANGELOG | medium | 01,02 |

Depends on all feature epics being in.
