# fynance — developer brief (for Claude Code)

This folder is an **orientation pack written for Claude Code** (not end users). Its
job is to give an agent a fast, faithful overview of the repository: what exists,
how it fits together, why it was built that way, and what is and isn't done.
End-user docs live in `doc/source/` (Sphinx); the authoritative working rules
live in the repo-root `CLAUDE.md`.

> **Relationship to `CLAUDE.md`**: `CLAUDE.md` is the source of truth for
> *commands, the layer map, and the hard invariants you must not regress*. This
> folder is the *narrative and depth* around it — rationale, per-area detail,
> current status, testing methodology. When the two disagree, trust `CLAUDE.md`
> and fix this folder.

## Read in this order

1. [`01-overview.md`](01-overview.md) — what fynance is, the current state, the
   repo map, the subpackages.
2. [`02-architecture.md`](02-architecture.md) — the package layout, the
   Numba kernels (golden-value parity), the rolling/walk-forward pattern, the
   estimator→models pipeline.
3. [`03-decisions.md`](03-decisions.md) — the design choices and *why* (Numba over
   new Cython, PyTorch over Keras, the loss-function split, NumPy over polars),
   plus the running ADR journal.
4. [`04-subpackages.md`](04-subpackages.md) — the per-subpackage map, public API
   surface, and the stability/modernisation policy that governs each.
5. [`05-testing.md`](05-testing.md) — the testing layers (pytest + doctests +
   benchmarks), how to run each, and the conventions (no lookahead bias).
6. [`06-status.md`](06-status.md) — what's done, what's in progress, known gaps,
   tooling, and deferred work.

## Tools kept here

- [`plans/`](plans/) — **active plan trees** (durable, hierarchical task plans).
  Each roadmap item being worked on expands into a `plans/<epic>/` tree of a
  global map + precise leaf specs that drive `/plan` → `/execute-leaf` →
  `/finish-task`. See [`plans/README.md`](plans/README.md). **The plan trees are
  gitignored** (kept local — R&D / strategy stays private); the descriptive docs
  above, the roadmap (`07-roadmap.md`) and `plans/README.md` are tracked.

## Conventions for keeping this current

- This is descriptive, not aspirational: write what the repo **is**, not what it
  should become. Open work goes in `07-roadmap.md` (local) and its executable
  expansion in `plans/`; history stays in git/`CHANGELOG.md`; the *why* in
  `03-decisions.md`.
- Finished plan trees move to `_archive/plans/` (gitignored).
