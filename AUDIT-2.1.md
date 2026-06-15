# fynance — full repository audit (v2.1.1)

Audit date: 2026-06-16 · commit: `develop` @ v2.1.1 · scope: package, tests, docs,
coherence, object-by-object.

## 0. Snapshot

| Metric | Value |
|--------|-------|
| Package LOC | ~14,400 (11 subpackages) |
| Test LOC | ~4,600 · **499 tests** + doctests |
| Coverage | **85%** overall (numba `@njit` under-reports — see §5) |
| Gates | ruff ✅ · mypy (135 files) ✅ · interrogate **93.6%** ✅ · Sphinx `-W` ✅ |
| Build | pure-Python (Numba kernels, no Cython, no `setup.py`) |
| Public API | 142 names at top level |

Overall: **healthy, coherent, well-tested.** The 2.x refactor landed cleanly
(layered + `typing.Protocol` seams, numpy lingua franca, PyTorch confined to
`models`, Numba kernels). The findings below are concentrated in (a) one real
correctness bug, (b) the legacy `backtest` plotting stack, and (c) minor dead
code / debug leftovers.

---

## 1. Findings (by severity)

### 🔴 HIGH — correctness

**H1 · `ARMAX_GARCH` swaps `psi`/`theta`.**
`econometric_models.ARMAX_GARCH(y, x, phi, psi, theta, …)` calls the kernel as
`_armax_garch(y, x, phi, theta, psi, …)` — the kernel signature is
`(y, x, phi, psi, theta, …)`, so the **external-regressor coefficients (`psi`)
and the MA coefficients (`theta`) are applied to the wrong terms.** Pre-existing
in the former Cython; faithfully preserved during the numba port (E7.01) to keep
parity. **Not caught by tests** (no case with `psi ≠ theta`). The plain
`ARMA_GARCH` path is unaffected. → Fix the call order and add a `psi ≠ theta`
regression test. *(This is the one item I'd fix before relying on ARMAX.)*

### 🟠 MEDIUM — tech debt / hygiene

**M1 · Legacy `backtest` plotting stack.** `_basis_plot` (0% cov), `plot` (24%),
`plot_backtest` (22%), `plot_tools` (14%), `dynamic_plot_backtest` (36%),
`backtest_neural_net` (23%). It is **not dead** — it powers `RollMultiLayerPerceptron`
live-training viz and `LossSeries` — but it is conceptually superseded by
`fynance.plot`, is barely tested, contains debug `print()`s, and is the reason
**matplotlib is imported eagerly at `import fynance`** (the new `fynance.plot` is
lazy). → Decouple/lazy-load matplotlib in the rolling-NN path; consider porting
the live viz onto `fynance.plot` and retiring the stack in 2.2.

**M2 · Dead code in `_metrics_helpers`.** `_roll_annual_return_py` and
`_roll_annual_volatility_py` are marked *"Old function"*, unused, yet still in
`__all__`. → Remove (and from `__all__`).

**M3 · Debug `print()` in stable modules.**
`portfolio/allocation.py:455` `print(mat_cov)` in a `LinAlgError` branch (leftover
debug); `models/rolling.py:439` `print(txt, end='\r')` (training progress — should
be opt-in/logging). The other `print`s live in the legacy plot stack (M1).

### 🟡 LOW — polish

**L1 · `money_management.py`** (36% cov) — old module; only `iso_vol` is used
(by the legacy `plot_tools`). Audit for relevance; fold or test.
**L2 · `fy.backtest` is the subpackage**, not the engine function (the function is
`fynance.backtest.backtest`). Documented, but a mild footgun; the flat top-level
namespace (142 names) is large.
**L3 · Stale-ish doc pages** `backtest.tools.rst` / `backtest.plot_object.rst`
document the legacy stack (M1) — revisit when M1 is addressed.
**L4 · Docstring coverage 93.6%** — above the 80% gate; a few internal helpers
lack docstrings.

---

## 2. Subpackage-by-subpackage

### `core` (2 mod · 100%/87% cov) — ✅ excellent
`PriceSeries` (numpy-backed value object; compose-not-subclass; price↔return
identities, `pnl` causal, `.pipe`, `to_numpy`/`to_torch` lazy) + `protocols`
(`DataSource`/`FeatureTransform`/`SignalModel`/`Allocator`/`CostModel`/`Metric`,
`runtime_checkable`). Clean, fully tested. Minor: a few `PriceSeries` branches
(13%) untested.

### `data` (5 mod · 66–100%) — ✅ solid
`load` dispatcher + `CSVSource`/`ParquetSource`; `align`/`resample` (causal,
past-only ffill); `train_test_split`/`walk_forward` (no-lookahead, embargo/purge).
`align` 66% — resample/ohlc branches under-tested. Good causality tests.

### `features` (10 mod) — ✅ good (coverage caveat §5)
Indicators (RSI/MACD/Bollinger/CCI/HMA/ROC/realized-vol/skew/kurt/autocorr),
momentums (sma/wma/ema/smstd/wmstd/emstd — Numba), `scale` (z-score, roll-rank),
`stats` (accuracy/dir-accuracy/percent_positive/tail_ratio/z_score/mad), `filters`
(Kalman), `engineering` (multi-resolution/Granger/IncrementalMoments), `regime`
(k-means). `_metrics_helpers` holds the Numba metric kernels (+ M2 dead funcs).
`money_management` (L1).

### `metrics` (4 mod · 67–100%) — ✅ good
`ratios` (sharpe/sortino/calmar/diversified_ratio/annual_vol + roll_*),
`drawdown` (drawdown/mdd/roll_*), `returns` (annual_return/perf_*/roll_*),
`summary` (one-call panel + registry). The E7 NaN fix (`roll_annual_volatility`
buffer aliasing) is here and tested. `returns` 67% (perf_index/perf_returns
branches).

### `models` (14 + loss 7) — ✅ strong
`BaseNeuralNet` (fit/predict → SignalModel), MLP/RNN/GRU/LSTM/TCN/Transformer/
attention, `ensemble` (leak-free stacking), `econometric_models` (Numba ARMA/GARCH
— see **H1**), `rolling` (`_RollingBasis`, purged CV), `training` (EarlyStopping,
sample weights), losses (Sharpe/Sortino/Calmar/Omega/Directional/Hybrid, all
100%). `rolling` 65% (live-viz branches → M1). econometric 49% (numba §5).

### `signal` (2 mod) — ✅ clean
`sign`/`threshold`/`rank`/`vol_target_position` + `SignalPipeline`. Causal, tested.

### `portfolio` (2 mod · 96%) — ✅ solid
`allocation` (ERC/HRP/IVP/MDP/MVP/MVP_uc/rolling_allocation), `sizing`
(kelly/vol_target/transaction_cost). One debug print (M3).

### `backtest` (engine 96% · result 96% · cost 100% · legacy plots low) — mixed
**New** (vectorized `backtest`, `BacktestResult`, `ProportionalCost`): excellent,
causal, well-tested. **Legacy** plot stack: M1.

### `plot` (4 mod · 100%/91%) — ✅ clean
Composable figures + `tearsheet`/`tearsheet_text`, lazy matplotlib, headless.

### `strategy` (1 mod) — ✅ clean
`Strategy` (compose features→model→signal→cost→backtest) + `run_walk_forward`
(no-lookahead, strengthened test).

### `estimator` (1 mod · 100%) — ✅
`target_function`/`_loglikelihood` (Numba ARMA/GARCH); `estimation` is an
explicit `NotImplementedError` placeholder (documented).

---

## 3. Cross-cutting coherence — ✅
- **Architecture invariants** hold: numpy at every seam, PyTorch only in `models`,
  Numba for kernels, protocols for composition, `Strategy` optional.
- **Causality / no-lookahead**: enforced + tested (engine shift, walk_forward/
  split embargo, feature property tests, walk-forward leakage probe).
- **Parity**: every Numba kernel has a golden/reference parity test (1e-9/1e-10,
  several bit-identical).
- **Naming**: consistent post-2.0 layout; no broken CLAUDE.md references.

## 4. Tests — ✅ with gaps
499 tests + doctests; strong on causality, parity, protocol conformance. Gaps:
legacy plot stack (M1), `align` resample, `money_management` (L1), and the
**missing `ARMAX_GARCH` psi≠theta test that hid H1**.

## 5. Coverage caveat (important)
Reported 85% **under-counts** the Numba modules: `coverage.py` cannot trace into
`@njit`-compiled bodies, so `momentums` (42%), `roll_functions` (42%),
`_metrics_helpers` (33%), `econometric_models` (49%) look low but are exercised by
the parity tests. The **genuine** low-coverage is the legacy `backtest` plot stack
(M1) and `money_management` (L1).

---

## 6. Recommended actions (priority order)
1. **Fix H1** (`ARMAX_GARCH` psi/theta) + regression test. *(correctness)*
2. **M2** remove dead `_roll_annual_*_py`. *(quick)*
3. **M3** drop the `allocation` debug `print`; gate the rolling `print`.
4. **M1** lazy-load matplotlib / decouple the rolling-NN live viz; plan retiring
   the legacy plot stack onto `fynance.plot` (2.2).
5. **L1** triage `money_management`.

None of these block current use of the core pipeline; H1 only affects the
ARMAX (external-regressor) GARCH variant.
