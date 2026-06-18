# 3 — Design decisions & rationale

The *why* behind the structure. Each entry is a choice that shapes the code; if
you're about to change one, read the rationale first. The prose below is the
*settled* rationale (standing decisions); the **Decision journal** at the bottom
is the running, dated ADR log.

## Numerical core

- **Numba `@njit` for all numerical kernels — no Cython.** The former `*_cy.pyx`
  kernels (features `momentums`/`metrics`/`roll_functions`, the ARMA/GARCH
  `estimator`/`econometric_models`) were ported to Numba `@njit` in the Python
  modules (E7 / 2.1.0); the build is now pure-Python with no compile step.
  Rationale: `@njit` gives near-C speed without a `.pyx`/`.c` pair to maintain or a
  C toolchain to ship. New Cython is only acceptable when wrapping a C library.
  See the dated ADR entry below — this tombstones the earlier "Cython extend-only"
  decision.
- **NumPy at the core; polars only at the I/O edges (pandas removed).** The
  linear-algebra-heavy core (allocation, rolling windows) is raw NumPy — fastest
  and the natural input for torch. Array-like *inputs* are accepted as
  numpy / torch / **polars** and immediately coerced to numpy/torch; table-shaped
  *outputs* are plain numpy (e.g. a structured array for the training log). See
  the dated ADR entry below — this reverses an earlier polars rejection.

## ML modernisation

- **PyTorch is the only ML backend.** Keras/TensorFlow code is retired, not
  extended. New architectures (LSTM/TCN/Transformer over the rolling MLP) and all
  training target `torch`.
- **Loss functions: two independent paths (the "Option C" split).** Evaluation/
  backtest metrics (Sharpe/Sortino/…) stay as NumPy formulas in
  `fynance.metrics`; the *training* losses are re-implemented as pure torch
  ops in `models/loss/`. The two paths never convert numpy↔torch — each is native
  to its context. Cost: the formula exists twice; benefit: no autograd-breaking
  conversions, no numpy in the training graph.

## Causality (the core invariant)

- **No lookahead, structurally.** Every feature at `t` is `f(data[..t])`; every
  training window trains on the strict past via the `_RollingBasis` iterator. This
  isn't a style preference — a single leaked future value invalidates a backtest.
  New features get a causal audit; tests forbid shuffling / future leakage.

## Build & packaging

- **`pyproject.toml` is the authoritative (pure-Python) build config; there is no
  `setup.py` and no compile step.** Single static `version` in `pyproject.toml`
  (one source of truth for `/release`). A pure-Python universal wheel
  (`py3-none-any`) + an sdist are built and trusted-published to PyPI on a `v*` tag.

## Decision journal (ADR)

Append-only, dated log of choices made since the dev-loop was set up — fed by
`/finish-task` (accepted) and `/abandon-task` (rejected / tombstone). The prose
above is the *settled* rationale; this journal is the *running* one. **Newest
first.**

Conventions:
- One entry per significant choice; skip the trivial (those live in
  git/`CHANGELOG.md`).
- `[tombstone]` = a feature was removed. Keep one line on *why it's gone* here and
  **purge its implementation rationale from the prose above** — negative knowledge
  so it isn't silently re-added later.

Template:
```
### YYYY-MM-DD — <short title> (PR #NN)  [accepted|rejected|tombstone]
- **Choice**: …
- **Why**: …
- **Rejected alternatives**: …
```

<!-- new entries below, newest first -->

### 2026-06-18 — Anti-churn brick 2: inference-time turnover mappers (PR #170)  [accepted]
- **Choice**: add three causal, composable position mappers to `fynance.signal` —
  `ema_smooth`, `deadband` (sticky/hysteretic, magnitude-preserving), `min_hold`
  (minimum dwell). The inference-time complement to brick 1's train-time penalty.
- **Why**: two layers beat one. The net-of-cost objective makes the model *want*
  to hold; the mappers give a hard, model-agnostic turnover cap on top (and work
  for rule-based strategies too). Kept as separate small functions (composable)
  rather than one mega-mapper.
- **Rejected alternatives**: folding everything into the existing `threshold`
  (it discards magnitude and is stateless — wrong tool); only the train-time
  penalty (can't hard-cap trade frequency).

### 2026-06-18 — Anti-churn brick 1: net-of-cost objective training (PR #169)  [accepted]
- **Choice**: penalize turnover **inside the training objective** — `ObjectiveModel`
  optimizes `Sharpe(positions*returns − cost*|Δpositions|)` via a new `cost` param —
  rather than only post-processing positions or relying on the backtest cost.
- **Why**: at realistic crypto fees (Kraken ~0.26% taker → ~0.52% round-trip) a
  high-frequency strategy that flips often is structurally unprofitable. Teaching
  the net the cost at train time makes it *hold* (the gradient couples positions
  across bars), which a post-hoc filter cannot do as well. Generic, opt-in
  (`cost=0` default = unchanged).
- **Rejected alternatives**: only an inference-time smoother/deadband (necessary
  but insufficient — the model still *wants* to churn); a separate cost-aware loss
  class (more API surface than threading `cost` through the existing objective).

### 2026-06-17 — Self-describing experiments (provenance block) (PR #163)  [accepted]
- **Choice**: provenance (what data + what features + run config produced a result)
  is recorded **generically in `fynance.research`** — `run_experiment` builds a
  structured `spec` and `write_report` renders a Provenance table — rather than as
  a per-repo manifest convention downstream.
- **Why**: the pain ("can't see the data/features behind a result") is generic, so
  the fix belongs in the tool: every consumer gets self-describing artifacts for
  free, with no convention to drift. Backward compatible (optional fields; old
  specs still render), so no break to existing `experiment.json`.
- **Rejected alternatives**: a hand-maintained "strategy card" / manifest in the
  private repo (manual, non-reusable, drifts); doing nothing (the result artifacts
  stay opaque).

### 2026-06-16 — Fix RTD build + remove the GitHub Pages docs workflow (PR #131)  [accepted]
- **Choice**: drop the `post_install: python setup.py build_ext --inplace` job
  from `.readthedocs.yaml`, and delete the master-only `.github/workflows/docs.yml`
  (Sphinx build + GitHub Pages deploy). Read the Docs remains the single docs host;
  the required CI `Docs (Sphinx, -W)` gate in `ci.yml` validates the build.
- **Why**: both still ran `setup.py build_ext`, which has not existed since v2.1.0
  (pure-Python/Numba) — so **RTD builds and the master Pages workflow had been
  failing on every build/push**. `docs.yml` also contradicted the RTD-canonical
  decision below.
- **Corrects** the 2026-06-14 entry: `.readthedocs.yaml` no longer "compiles the
  Cython ext", and a GitHub Pages deploy workflow *did* exist (master-only) and is
  now removed.
- **Note**: per-version (per-tag) docs are an RTD **dashboard** setting (Admin →
  Versions / Automation Rules), not something the repo config controls.

### 2026-06-15 — All Cython ported to Numba; pure-Python build (E7 / 2.1.0)  [tombstone]
- **Choice**: remove **all** Cython. The `*_cy.pyx`/`.c` kernels — features
  `momentums`/`metrics`/`roll_functions` and the ARMA/GARCH `econometric_models`/
  `estimator` — were ported to **Numba `@njit`** functions living in the `.py`
  modules; `setup.py` and the whole compile step were deleted. The build is now
  pure-Python and `release.yml` ships a single `py3-none-any` wheel (no
  `cibuildwheel`/manylinux).
- **Why**: one implementation per kernel instead of a `.pyx`/`.c`/Python triple;
  no C toolchain to ship or maintain; `@njit` matches the former Cython speed.
  Parity was captured as golden values from the compiled `.so` **before** deletion
  and asserted to 1e-9/1e-10 — this strictness surfaced two real bugs
  (`roll_annual_volatility` buffer aliasing, a missing `roll_annual_return` term).
- **Tombstones**: the earlier "Cython for the existing hot paths, kept extend-only"
  and "`setup.py` only compiles Cython / `cibuildwheel` manylinux wheels"
  decisions — purged from the prose above.
- **Rejected alternatives**: keeping the Cython twins (the maintenance/build cost
  the port removes); a C-extension rewrite (no payoff over `@njit`).

### 2026-06-14 — Read the Docs is the canonical docs host (roadmap §3.1)  [accepted]
- **Choice**: keep **Read the Docs** as the single documentation host. `.readthedocs.yaml`
  already builds the site (compiles the Cython ext, installs `.[doc]`, runs the
  Sphinx config), the README badge + links point to `fynance.readthedocs.io`, and
  there is **no** GitHub Pages deploy workflow. RTD now also builds with
  `fail_on_warning: true`, matching the CI `sphinx-build -W` gate.
- **Why**: RTD was already the de-facto host; the earlier "GitHub Pages today /
  RTD migration open" note was stale. No migration needed — just confirm RTD and
  align the warning policy.
- **Rejected alternatives**: GitHub Pages (never actually wired up); a custom
  domain (not needed).

### 2026-06-14 — Track doc/dev + CLAUDE.md publicly, mirroring dccd (PR #62)  [accepted]
- **Choice**: stop gitignoring `doc/dev`. The descriptive pack `01–07` (including
  the roadmap), `README.md`, `plans/README.md` **and `CLAUDE.md`** are now tracked,
  mirroring dccd. Only the plan trees (`doc/dev/plans/<epic>/`), any `_archive`
  snapshot and the Claude harness settings (`.claude/`) stay local.
- **Why**: the `.gitignore` `dev/` rule (meant for a scratch dir) was accidentally
  swallowing **all** of `doc/dev/`, so the "docs of record" the dev loop writes to
  (this ADR journal, the status file) were never actually shared — directly
  contradicting CLAUDE.md, which claimed `01–06` were tracked. dccd publishes the
  full pack + roadmap + `CLAUDE.md`; the roadmap was reviewed and holds only
  engineering tasks (TCN / Transformer / refactors), no secrets or proprietary
  strategy, so the earlier privacy concern does not apply.
- **Reverses**: the 2026-06-13 decision to keep the roadmap private and reject full
  dccd parity — superseded; full dccd parity adopted at maintainer request.

### 2026-06-14 — Replace pandas with polars at the edges, numpy at the core (PR #61)  [accepted]
- **Choice**: drop the `pandas` dependency. Inputs that accepted pandas
  (`BaseNeuralNet.set_data` / `_set_data`, `econometric_models.MA`) now accept
  **polars** (plus numpy/torch), coerced via `to_numpy()`. Outputs that returned
  pandas now return plain **numpy**: `rolling_allocation` → `(ndarray, ndarray)`,
  `RollMultiLayerPerceptron.get_stats` → a structured `ndarray`. Core computation
  stays numpy.
- **Why**: a single torch/numpy-native data path; pandas only ever lived at the
  I/O edges. polars is lighter and the maintainer's preferred frame. The risky
  piece — `rolling_allocation` (previously untested) — was rewritten pandas-free
  in numpy and verified for **exact parity** against the old pandas implementation
  across MVP/ERC/IVP/HRP (various `ret`/`drift`) before pandas was removed; a
  golden-value regression test now guards it.
- **Reverses**: the earlier "polars POC evaluated and rejected; pandas at the
  edges" rationale (purged from the prose above).
- **Note**: technically API-breaking for the 1.x frozen API (return types
  pandas → numpy) — flagged in `CHANGELOG.md`; accepted per maintainer direction.

### 2026-06-13 — Adopt the tracked dev-loop docs (doc/dev) with a private roadmap  [accepted]
- **Choice**: mirror the dccd tooled loop — `doc/dev/{01..06}` descriptive docs +
  an ADR journal + a `06-status` + plan trees, wired through
  `.claude/workflow.json` (`decisions`/`status`/`plans_dir`). The **roadmap
  (`07-roadmap.md`) and the plan trees stay gitignored** (local only), continuing
  the existing choice to keep `TODO.md` out of git.
- **Why**: `/finish-task` and `/groom-docs` need real target docs (an ADR journal
  and a status file) to write to; without them the loop ran half-blind. Keeping
  the roadmap private preserves the prior posture — the R&D section holds strategy
  ideas not meant for a public repo. The descriptive docs carry no secrets, so
  they're tracked.
- **Rejected alternatives**: full public parity with dccd (would publish the R&D
  roadmap — rejected on privacy); doing nothing (leaves `/finish-task`'s ADR/status
  steps with nowhere to write).

### 2026-06-15 — fynance 2.0: layered architecture + protocols (D4: Strategy API)  [accepted]
- **Choice**: the 2.0 refactor adopts a **layered architecture with `typing.Protocol`
  seams** (`DataSource`, `FeatureTransform`, `SignalModel`, `Allocator`,
  `CostModel`, `Metric`), ports&adapters **only** at I/O (`fynance.data`), **not**
  full hexagonal. numpy is the lingua franca; pytorch stays confined to
  `fynance.models`; numba for hot loops. New maillons: `core` (`PriceSeries`),
  `data`, `metrics` (out of `features`), vectorized `backtest`, `signal`,
  `portfolio` (ex-`algorithms`), `plot`/`tearsheet`, `strategy`.
- **D4 — Strategy API**: a **fluent dataclass-style** constructor (protocol-typed
  slots as keyword args) with `.run` / `.run_walk_forward`, **not** a declarative
  config tree. Every slot is optional so the maillons stay usable standalone
  (composition, not a forced pipeline).
- **Why**: the domain is maths on arrays — full hexagonal adds ceremony with no
  payoff; protocol composition gives the modularity/testability where it matters
  while keeping the toolbox usable piecewise. Fluent slots match the scientific
  Python idiom (sklearn-like) and keep the orchestrator optional.
- **Note**: breaking 2.0 (no compat shims): `algorithms`→`portfolio`,
  perf-metrics `features`→`metrics`. Flagged for the major bump in `CHANGELOG.md`.
