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

### 2026-07-06 — GARCH family epic (PRs #255, epic)  [accepted]

- **Choice**: extend the volatility family inside
  `models.econometric_models` (single-implementation policy): GJR-GARCH and
  EGARCH Numba filters + `loglik_garch` with Gaussian/Student-t innovations
  (kernels return `-inf` on inadmissible regions instead of raising); the
  scipy MLE driver, `VolatilityResult` (std errors, AIC/BIC, forecast,
  simulate) and the `features/garch.py` `model=`/`dist=` passthrough land as
  a second leaf on top.
- **Why**: GARCH(1,1)-normal systematically underestimates tail vol on
  equity-like series; `estimator.estimation()` was an explicit stub with no
  working MLE path anywhere; asymmetry (GJR/EGARCH) and fat tails
  (Student-t) are the two standard remedies.
- **Rejected alternatives**: depending on `arch` (heavy, duplicates the
  house kernels); exceptions inside kernels (breaks nopython + optimizer
  workflows); a separate garch subpackage (violates the no-duplication
  policy).

### 2026-07-06 — Metrics & trade analytics epic (PRs #253–#254, epic)  [accepted]

- **Choice**: four additive metric families — benchmark-relative two-curve
  metrics (`metrics.benchmark`, kept OUT of the scalar `METRICS` registry:
  they take `(X, B)`); tail-risk (`metrics.risk`: VaR/CVaR/CDaR with
  historical/gaussian/Cornish-Fisher estimators, registered in `summary()`);
  turnover/exposure analytics (`metrics.trading` + optional tearsheet panel
  behind a default-off kwarg); trade-level round-trips (`metrics.trades`
  structured arrays + `BacktestResult` conveniences).
- **Why**: the map's bluntest metric gaps — no two-curve function at all,
  no VaR/ES, no per-trade statistics; empyrical/pyfolio/vectorbt set the
  practitioner expectation.
- **Rejected alternatives**: registering two-input metrics in the scalar
  registry (breaks the `summary()` contract); pandas-based trade logs
  (structured numpy arrays match the house labels.py precedent);
  interpolated historical quantiles (deterministic order statistic keeps
  hand tests exact and tail_dependence symmetric).

### 2026-07-06 — Anti-overfitting guards epic (PRs #249–#252, epic)  [accepted]

- **Choice**: complete the overfitting-guard triad as four parallel bricks —
  purged walk-forward HP search (`models.tuning`, grid/random only) exposing
  `n_trials` straight into `deflated_sharpe_ratio`; CSCV/PBO diagnostic
  (`research.overfit`); block/stationary bootstrap (`research.bootstrap`,
  dependence-preserving null as a drop-in analogue of `permutation_test`);
  CPCV splitter in `data.split`.
- **Why**: the guards module had permutation + DSR but nothing diagnosing
  selection bias across a *family* of tried configs, and every HP sweep was
  a hand-rolled leak-prone loop hiding its trial count; the iid permutation
  null destroys autocorrelation.
- **Rejected alternatives**: an optuna dependency (grid/random suffice; no
  heavy dep); modifying `permutation_test` in place (the block variant is
  additive, old semantics preserved); White's Reality Check / SPA (PBO is
  the assumption-light standard and fits the Ledger data we already have).

### 2026-07-05 — Cross-sectional factor research epic (PRs #243–#248, epic)  [accepted]

Gives fynance the *input half* of the factor workflow the v2.11 panel harness
consumes. Six parallel leaves, key choices:

- **Choice**: pairwise rolling kernels (`roll_cov/corr/beta`, `cross_corr`)
  live in `features.roll_functions` (not metrics) so the future
  benchmark-relative metrics can reuse them; windows are trailing and include
  `t` (house convention, matches `roll_min`); cross-sectional operators
  (`cs_*`) are NaN-aware per bar (panels with dead assets are the norm);
  AFML labels (`triple_barrier`, `meta_labels`, uniqueness weights) are
  documented as **training targets that read future prices by design** — to
  be consumed only through purged splits; MDA importance fits once per fold
  and permutes the test window only (no per-feature refit); the factor suite
  splits metrics (`metrics.factor`) from figures (`plot.factor`) to keep
  `import fynance` matplotlib-free.
- **Why**: the mapped gap was blunt — everything was single-asset
  time-series; the panel training stack (`ObjectiveModel`, `RankingLoss`,
  per-bar IC) had no tooling to build/evaluate the factor panels it consumes.
- **Rejected alternatives**: depending on alphalens (abandoned upstream,
  pandas-first); pandas-based implementations (numpy is the lingua franca);
  putting labels in `data/` (they are feature-layer transforms of prices).

### 2026-07-05 — Constraint overlay as least-distance projection (PR #239)  [accepted]

- **Choice**: enforce book-level limits (box / gross / net / group) as a
  *post-hoc least-distance projection* (`portfolio.constraints.project_weights`,
  SLSQP on the split `v = p - m` so `sum(|v|)` becomes linear), with a fast
  clip-and-scale path when only box+gross are active, rather than baking the
  constraints into each allocator's optimizer.
- **Why**: an overlay composes with *any* weight source (allocators, signal
  mappers, `ObjectiveModel` books) and keeps the frozen allocation API
  untouched; the split formulation keeps the QP smooth for SLSQP (already the
  house optimizer); the fast path covers the common case ~322x faster
  (measured on a (500, 12) book).
- **Rejected alternatives**: per-allocator constraint parameters (N x API
  surface, still leaves signal-side books unconstrained); a dedicated QP
  solver dependency like cvxpy/osqp (heavy dependency for one projection);
  exact soft-thresholding closed form for the gross cap (only valid without
  box/net/group interactions).

### 2026-07-05 — Risk budgeting as a separate `RBP` allocator (PR #238)  [accepted]

- **Choice**: ship risk budgeting as a new `RBP(X, budgets=None, cov=None)`
  function with the Roncalli least-squares objective
  `sum_i (w_i (Sigma w)_i - b_i w'Sigma w)^2`, sharing ERC's scaffolding
  (unit-trace rescale, SLSQP, bound clamping) — not as a `budgets=` parameter
  on `ERC`.
- **Why**: `portfolio.allocation` is a frozen API — a new function is purely
  additive and leaves ERC's contract untouched; `budgets=None` reproduces ERC
  within optimizer tolerance (regression-tested at 1e-4), and budget matching
  is verified through `portfolio.attribution` (ex-ante error ~8e-7 on
  synthetic panels, ~7e-4 through the rolling path).
- **Rejected alternatives**: extending `ERC`'s signature (behavior-drift risk
  in the stable API); Roncalli's log-barrier / fixed-point formulation
  (cleaner convexity but a second optimizer pattern to maintain for no
  observed accuracy gain here).

### 2026-07-05 — Conditioned covariance estimators as interchangeable callables (PR #235)  [accepted]

- **Choice**: ship covariance conditioning as a standalone
  `portfolio.covariance` module of `(T, N) -> (N, N)` callables — closed-form
  Ledoit-Wolf (three targets), RiskMetrics EWMA on a Numba kernel, PCA factor
  model, Marchenko-Pastur clipping — rather than baking estimators into each
  allocator. The allocators gain an opt-in `cov=` callable (landed in PR #236),
  `None` keeping today's `np.cov` path bit-for-bit; `MDP`'s `cov=` path
  evaluates the diversification ratio from the fixed matrix instead of
  recomputing `np.cov` per optimizer iteration.
- **Why**: raw `np.cov` on short windows is the dominant source of MVP/ERC
  weight instability; the closed forms need numpy only (no new dependency);
  a callable seam preserves the frozen `portfolio.allocation` API and makes
  the estimators reusable elsewhere (the `portfolio.attribution` diagnostics
  landed in PR #237 on the same seam, validating that ERC equalizes
  contributions).
  Verified OOS on synthetic two-block panels: condition number 97.4 -> 91.1
  (LW) / 82.5 (denoised); mean MVP OOS vol 0.005909 -> 0.005790 over 20 seeds.
- **Rejected alternatives**: depending on scikit-learn (heavy dependency for a
  ~40-line closed form); nonlinear/oracle shrinkage (unwarranted complexity —
  can land later as just another callable); per-allocator estimator flags
  (couples the stable API to an estimator zoo).
- **Epic closed** in PR #240 with `book_vol_target` (the multi-asset
  counterpart of `vol_target`, `W[t-1] -> r[t]` timing), completing roadmap
  §3 (covariance seam #235/#236, attribution #237, `RBP` #238, constraint
  overlay #239).

### 2026-06-23 — Multi-asset / panel R&D harness (PRs #215–#219, epic)  [accepted]

Took the research harness from single-asset to **multi-pair**: a model predicts a
position **book** `(T, N)`, a rule allocates, the book is backtested and reported
with per-asset attribution. Five atomic leaves: #216 (IC), #215 (`ObjectiveModel`
panel), #217 (book-aware losses), #218 (walk-forward + attribution), #219 (book
report). Test count 946 (with doctests). Decisions worth recording:

- **Book aggregation lives in the model *and* (additively) in the losses.**
  `ObjectiveModel` sums the per-asset net returns to the 1-D book return before
  the loss, so the existing 1-D losses keep working unchanged; the ratio losses
  were *also* made book-aware (sum a `(T, N)` input across assets) so a custom
  architecture can feed per-asset returns directly. Both reduce to the same
  scalar, and `(T, 1)`/1-D are bit-identical to before (verified: `N=1`
  max-abs-diff `0.0`).
- **A 3-D panel `X (T, N, M)` is flattened to `(T, N·M)`** for the default dense
  net, with `N` remembered for the `Linear(dim, N)` head; a custom `net=` receives
  the flattened matrix (3-D pass-through to a custom net is deferred).
- **Per-asset attribution is gross-contribution only.** `BacktestResult`
  carries an optional `asset_gross_returns` `(T, N)` (summing to the book gross
  return); the cost model already returns book-aggregated turnover, so per-asset
  *net* costs are not split — turnover is shown per asset from the position book.
  `None` for a single-asset run (strict back-compat).
- **IC defaults to cross-sectional-per-bar** for a `(T, N)` panel (the ranking
  use-case; `axis=1` gives the per-asset time-series IC), and **horizon labels are
  non-overlapping by default** (overlapping H-bar labels inflate the IC).
- **Rejected alternatives**: re-touching `ObjectiveModel` to pass per-asset returns
  to the loss (kept the model's pre-aggregation, made losses additively book-aware
  instead); extending `BacktestResult` with per-asset *net* costs (out of scope —
  the cost model is book-level).

### 2026-06-22 — Audit round 2: regression + residual bugs (PRs #201–#207, v2.10.1)  [accepted]

A second audit pass after v2.10.0 caught **one regression** the first remediation
introduced — `RollMultiLayerPerceptron.run(backtest_kpi=True)` (the default path)
raised `IndexError` because `__next__` bumps `self.i` past the last filled slot
before `StopIteration`; fixed by clamping the KPI index — plus residual
pre-existing bugs the first pass missed. Shipped in 7 atomic PRs (parallel
worktrees). Test count 880 (with doctests). Two decisions worth recording:

- **`RNN`/`GRU`/`LSTM` are now drop-in `SignalModel`s via a zero-initialized
  state.** Since these are stateless gated cells (each row independent), `H`/`C`
  carry no meaning across a call, so `fit(X, y)`/`predict(X)` default the state to
  zeros; the explicit-state forms remain for callers that thread state. Chosen
  over making them raise (they advertised `SignalModel` but `.fit` previously
  `TypeError`'d).
- **Training losses saturate smoothly, not by a hard clamp.** The v2.10.0
  `MAX_RATIO` clamp made the loss finite but zeroed the gradient on low-risk
  batches; replaced with `MAX_RATIO * tanh(ratio / MAX_RATIO)` + a scale-invariant
  floor so a residual gradient always survives. **Lesson: parallel-worktree
  agents must not use `git stash` (shared stash stack collides) — verify
  fail-before via a temp copy.**


### 2026-06-22 — Full code audit + remediation (PRs #188–#196)  [accepted]

A deep correctness/tests/docs audit (the 5 gates were already green) found logic
bugs, weak/missing tests and doc drift the gates miss. Remediated in 9 atomic
PRs (one concern each, parallel worktrees): features `axis=1` path, portfolio
optimizers, metrics/econometrics, models training+losses, NN base/recurrent
contract, data/core/signal/backtest, research guards, and doc drift. Roadmap §1
held the backlog; closed here. Test count 658 → 819.

Two choices worth recording as standing knowledge:

- **`RNN`/`GRU`/`LSTM` are *not* temporal recurrent nets — kept as stateless
  gated feed-forward cells and documented as such.** They feed the time axis to
  `nn.Linear` as the batch dim; no state crosses time. Implementing genuine
  recurrence (a `(T, S, N)` loop carrying hidden state) would change `forward`,
  batch semantics and `set_data`, rippling into `rolling`/`objective`. We chose
  the low-risk correct path: remove the false `(T, S, N)` contract, fix the
  concrete bugs, and pin the stateless behavior with a test. **Negative
  knowledge: do not "fix" these into recurrence casually — it is a deliberate,
  separable enhancement, not a bug.**
- **Optimizer-scale invariants made explicit.** ERC/MVP_uc rescale the
  covariance to unit trace before SLSQP (the objective otherwise sinks below
  `ftol` on return-scale data and the solver returns the 1/N start); the result
  is argmin-invariant. Zero-denominator ratio conventions were unified
  (`calmar` → `+inf` like `sharpe`).


### 2026-06-21 — Library-bricks epic: roadmap §1 shipped (PRs #177–#182)  [accepted]
- **Choice**: ship the data-agnostic "library bricks" (roadmap §1) as six atomic
  PRs and **re-scope** the roadmap — real-data strategy research moves to the
  private `fynance-research` repo; the public roadmap keeps only reusable,
  data-agnostic library work. Bricks: `core.OHLCV`, OHLCV indicators, causal
  GARCH feature, `RegimeMoE`, adaptive windows, non-linear cost (each entry below).
- **Why**: most remaining roadmap items were blocked on real data, not on missing
  code; separating the *library* (here) from the *research* (private repo) lets the
  public package stay data-agnostic and result-free while still gaining the bricks
  the research needs. One PR per concern keeps each disposable and reviewable.
- **Rejected alternatives**: one fourre-tout branch (against the one-PR-one-concern
  rule); keeping the data-blocked research items on the public roadmap (drift —
  they never move without data that lives elsewhere).

### 2026-06-21 — `core.OHLCV` value object for the multi-series input (PR #177)  [accepted]
- **Choice**: introduce a thin numpy-backed `OHLCV` value object (composition, not
  `ndarray` subclassing — mirroring `PriceSeries`) as the input contract for the
  multi-series indicators, rather than passing loose `high`/`low`/`close`/`volume`
  arrays everywhere. `close` required, others optional (absent-field access raises),
  equal length enforced at construction.
- **Why**: a typed, validated, reusable container catches misaligned/missing series
  once at the edge instead of in every indicator; the indicators still accept raw
  arrays too, so the object is an *option*, not a tax.
- **Rejected alternatives**: a pandas/polars DataFrame as the contract (heavier,
  re-introduces a frame dependency at the core); subclassing `ndarray` (the
  fragility `PriceSeries` already avoids).

### 2026-06-21 — OHLCV indicators: raw-arrays-first API with an OHLCV overload (PR #182)  [accepted]
- **Choice**: implement `atr`/`adx`/`williams_r`/`obv`/`vwap` as functions taking
  the **raw aligned arrays** as the primary signature, with a thin dispatch that
  also accepts a single `OHLCV` as the first argument. Rolling loops are Numba
  `@njit` kernels; ATR/ADX use Wilder smoothing.
- **Why**: raw-arrays-first matches the existing `features` idiom (every other
  indicator takes arrays), keeps the functions usable without constructing a
  container, and the overload gives the typed path for free.
- **Rejected alternatives**: OHLCV-only signatures (forces object construction,
  breaks the array idiom); methods on `OHLCV` (couples the container to the whole
  indicator library).

### 2026-06-21 — Causal GARCH feature = expanding-fit + forward-filter (PR #179)  [accepted]
- **Choice**: expose GARCH(1,1) conditional volatility as a feature by fitting the
  parameters on a training prefix (optionally refit on the expanding window) and
  **forward-filtering** σ over the series; the `min_train` warmup is `NaN`. Reuse
  the authoritative `models.econometric_models` recursion + `estimator` likelihood;
  the MLE fit is a thin scipy `SLSQP` wrapper.
- **Why**: σ_t in GARCH is 𝓕ₜ₋₁-measurable, so filtering forward with past-fit
  parameters is strictly causal — the only leak risks are the parameters (handled
  by expanding-fit) and the warmup (NaN'd). Reusing the single recursion respects
  the estimator no-duplication policy.
- **Rejected alternatives**: a single in-sample fit over the whole series (peeks —
  leaky); re-implementing the GARCH recursion in `features` (duplicates parameter
  logic the policy forbids).

### 2026-06-21 — Regime-conditioned architecture: causal detector + MoE (PR #180)  [accepted]
- **Choice**: `RegimeMoE` routes an objective-aligned net by the **causal** regime
  (`RegimeDetector` fit-on-train / assign-online, from a designated price/level
  column of `X`). Default `routing="soft"` (a learned regime embedding concatenated
  to the features through a shared trunk, differentiable end-to-end); `routing="hard"`
  offers one expert per regime. Reuses `ObjectiveModel` for training (objective /
  mini-batch / cost / seed).
- **Why**: conditioning needs a *causal* regime to be backtest-honest; the
  in-sample `detect_regimes` would leak. Soft routing trains end-to-end and shares
  data across regimes (more sample-efficient than fully separate experts) while
  hard routing stays available when regimes are believed truly disjoint. Composing
  `ObjectiveModel` avoids duplicating the training loop.
- **Rejected alternatives**: routing on in-sample `detect_regimes` (leaky); a
  bespoke training loop (duplicates `ObjectiveModel`); hard routing as the default
  (less sample-efficient, non-differentiable gate).

### 2026-06-21 — Non-linear cost = square-root market-impact law (PR #178)  [accepted]
- **Choice**: model convex execution cost as `fee*turnover + impact*turnover**1.5`
  (square-root impact law) in `MarketImpactCost`, reusing the `ProportionalCost`
  turnover definition so `exponent=1, impact=0` collapses exactly to the linear model.
- **Why**: real impact grows super-linearly with trade size; the √-law is the
  standard, single-parameter convex form, and making the linear case an exact
  special case keeps it a drop-in extension of the existing cost model.
- **Rejected alternatives**: a full order-book / market-impact simulator (out of
  scope for a vectorized backtest); a fixed quadratic only (√-law is the empirically
  standard default; `exponent` stays tunable).

### 2026-06-18 — Mini-batch training for ObjectiveModel (PR #173)  [accepted]
- **Choice**: add contiguous mini-batch SGD (`batch_size`/`shuffle`) to
  `ObjectiveModel`; a `fit` now does `epochs * ceil(T/batch_size)` steps instead of
  `epochs` full-batch steps. Chunks stay time-ordered so the turnover penalty
  remains meaningful (carried across chunk boundaries when not shuffled).
- **Why**: the full-batch path gave only `epochs` (~40) gradient updates on
  hundreds-of-thousands of bars — the minute-resolution models were drastically
  **under-trained**, which alone could explain weak/null results. Mini-batching is
  the prerequisite to honestly test whether OHLC carries edge. `batch_size=None`
  preserves the old behaviour (backward compatible).
- **Rejected alternatives**: only raising `epochs` (full-batch steps are O(T) each
  and still few); shuffling individual rows (breaks the turnover diff / Sharpe
  temporal structure) — hence *chunk-order* shuffle, rows kept ordered.

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
