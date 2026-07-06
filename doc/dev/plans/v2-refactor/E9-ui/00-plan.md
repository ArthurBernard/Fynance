---
plan: v2-refactor/E9-ui
kind: global
status: done
roadmap: "E9 ui — optional Streamlit playground (code a signal fn → see the tearsheet)"
release_on_done: false
---

# E9 — UI: Streamlit playground (optional, last)

Realize the "code on the left, perf on the right" vision at ~5% of an IDE's cost:
a thin Streamlit app over `tearsheet()`. **Optional extra** (`fynance[ui]`),
**outside** the importable package (`apps/`), built last. Not on the critical path.

## Leaves

| # | Leaf | Complexity | Deps |
|---|------|-----------|------|
| 01 | Streamlit playground app | medium | — |

Depends on E5 (tearsheet) and E8 (Strategy). May trail into 2.1.
