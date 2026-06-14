# Plan trees — durable, hierarchical task plans

This directory holds **active plan trees**: the file-based expansion of a roadmap
item into an executable plan. The `/plan`, `/execute-leaf` and `/finish-task`
skills read and write it. This `README.md` (the format reference) is **tracked**;
the actual plan trees are **gitignored** (local — they may contain R&D/strategy
detail), as is the roadmap they expand. Finished trees move to
`../_archive/plans/`.

## Why this exists

Plan mode writes to `~/.claude/plans/*.md` (outside the repo), so a plan is lost on
`/compact`, and nothing records *whether we planned one slice or the whole set*.
Plan trees fix both: plans are **files** (durable, reviewable) and **hierarchical**
(a global map + precise leaves).

## Layout

```
doc/dev/plans/<epic-slug>/
  00-plan.md            # global: goal, decomposition, leaf checklist, deps
  01-<leaf-slug>.md     # leaf: precise, agent-executable spec
  02-<leaf-slug>.md
  ...
```

**Depth is adaptive — never forced.**

- Trivial task → a *single* leaf file, **no global** `00-plan.md`.
- Normal task → a global `00-plan.md` + N leaves.
- A leaf still too big → its own sub-directory with a `00-plan.md` + sub-leaves
  (recursion). The deepest level must be precise enough that an agent executes it
  without re-deciding anything.

## Lifecycle

```
/pick-task → /plan (build tree)
  → /execute-leaf <epic> next   (model from the leaf's `complexity`; implement + test + verify)
  → /finish-task                (tests, ADR, PR, archive leaf, tick global)
  → … repeat per leaf (deps respected) …
  → last leaf → roadmap line removed, global done → /release
```

> Because the trees are gitignored here, there is **no "plan PR"** step (unlike a
> repo that tracks its plans): the tree lives locally and the work lands through
> the normal feature-branch PRs that `/finish-task` opens. To track plans in git,
> remove the `doc/dev/plans/*` lines from `.gitignore`.

## Frontmatter

### Global `00-plan.md`
```yaml
---
plan: <epic-slug>
kind: global
status: planning | executing | done
roadmap: "<verbatim roadmap line this expands>"
release_on_done: true
---
```

### Leaf `NN-<slug>.md`
```yaml
---
plan: <epic-slug>
kind: leaf
status: todo | doing | done
complexity: low | medium | high     # derives the execution model (haiku/sonnet/opus)
deps: []                            # leaf numbers that must finish first
parallel: false                     # may run concurrently with sibling leaves
---
```

A leaf body is a precise spec: files to touch, the change, how to test, and the
real-data/numerical check that proves it works.
