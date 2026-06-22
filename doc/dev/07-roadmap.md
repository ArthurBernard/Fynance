# 7 — Roadmap / next steps

This file is the **single source of truth** for open work — read by `/pick-task`,
updated by `/finish-task` / `/abandon-task`. **Tracked** (mirrors dccd). Finished
work is *removed* from here: git log + `CHANGELOG.md` are authoritative for *what*
shipped, `03-decisions.md` for *why*. Keep it short and true.

> Les numéros de section = état de travail, librement renumérotables. Ils
> n'apparaissent jamais dans les commits/CHANGELOG — la traçabilité passe par le
> numéro de PR. Tâche terminée = ligne supprimée (pas de section Done).
>
> Loop : `/pick-task` → `/plan` (arbre dans `plans/`) → `/execute-leaf` →
> `/finish-task`. Release : `/release` quand `[Unreleased]` est suffisamment rempli.

> **fynance 2.x livré** (v2.9.0 sur `master`/PyPI). Le refactor en couches, le port
> Cython→Numba (build pure-Python), le harnais R&D `fynance.research` (S1–S3), la
> brique d'entraînement aligné-objectif (`ObjectiveModel`) et les **bricks de
> librairie** (conteneur `OHLCV` + indicateurs ATR/ADX/Williams %R/OBV/VWAP, feature
> GARCH causale, `RegimeMoE`, fenêtres adaptatives, coût market-impact non-linéaire)
> sont **terminés** — voir `CHANGELOG.md` / `03-decisions.md`. Un **audit complet** (2026-06-22) a relevé des bugs de
> correctness, des trous de tests et des dérives de doc non couverts par les
> gates — backlog détaillé en **section 1** ci-dessous.

> ⚠️ **Hors scope ici** : la recherche de stratégie sur **vraie data** (benchmarks
> empiriques loss / architecture / normalisation, évaluation Sharpe out-of-sample,
> régimes en ligne, sélection de features) vit dans le **repo privé**
> `fynance-research` qui dépend de fynance. La roadmap publique ne tient que du
> code de librairie réutilisable, data-agnostique.

---

## 1. Remédiation audit — correctness / tests / docs (2026-06)

Audit complet du 2026-06-22 : les 5 gates (pytest/ruff/mypy/interrogate/sphinx)
passaient déjà ; les points ci-dessous sont des **bugs de logique**, des **trous
de tests** et des **dérives de doc** que les gates ne voient pas. Regroupés en PR
atomiques **parallélisables** (fichiers disjoints). Sévérité notée par point.

### 1.A — features : chemin 2-D `axis=1` cassé · `[fix/features-axis1]`
- [ ] **CRITICAL** `Scale.scale`/`Scale.revert` : la branche `axis==1` est calculée puis jetée (pas de `return`) → axis=1 silencieusement ignoré.
- [ ] **CRITICAL** `_wrappers.wrap_window` : fenêtre clampée sur `X.shape[0]` (axe séries) **avant** la transposition → `w > n_colonnes` rétréci en silence sur axis=1 (sma/wma/smstd/roll_*/z_score/cci/hma/rsi/roc/realized_volatility…).
- [ ] **CRITICAL** `momentums.z_score`/`roll_z_score` (kind='e') : conversion `α=1-2/(1+w)` appliquée **deux fois** sur axis=1.
- [ ] **HIGH** `_wrappers` : `np.AxisError` supprimé en NumPy 2 → utiliser `np.exceptions.AxisError` ; valider `axis < ndim` avant `shape[axis]`.
- [ ] **HIGH** `stats.mad(axis=1)` lève (broadcast) — câbler le wrapper `axis`.
- [ ] **HIGH (tests)** tous les asserts axis=1 utilisent un tableau `(6,1)` dégénéré → ajouter de vrais tests multi-colonnes (parité `f(X,axis=1)` vs per-row 1-D).
- [ ] **MEDIUM (docs)** formules `wma`/`wmstd`/`emstd` ne correspondent pas au code (normalisateur, terme de déviation).

### 1.B — features : guards & docs indicateurs · `[fix/features-guards]`
- [ ] **MEDIUM** `ohlcv.atr`/`adx` : pas de garde `w>=1` (ZeroDivisionError pour `w=0`).
- [ ] **MEDIUM** `ohlcv.williams_r` : doc annonce `[-100,0]` mais pas de clamp (borne conditionnelle à OHLC valide) — documenter ou clamper.
- [ ] **MEDIUM** `indicators.bollinger_band` : `DeprecationWarning` morte ("removed in fynance 2.0") émise à chaque appel — supprimer.
- [ ] **MEDIUM (docs)** `roll_functions.roll_min`/`roll_max` : math off-by-one (`w+1` vs fenêtre `w`) ; `roll_max` "See Also" se référence lui-même.
- [ ] **MEDIUM** `money_management.iso_vol` : rendement réciproque `s_{t-1}/s_t-1` au lieu de `s_t/s_{t-1}-1` ; + tests (causalité, cap, plancher).
- [ ] **LOW** `regime.regime_features` : ligne 0 `(0,0)` parasite injectée dans le clustering ; garde cluster vide.
- [ ] **LOW (tests)** `garch` : balayage causalité par-index + corriger commentaire trompeur ; borne `min_train`. `engineering` : granger lag>1, `IncrementalMoments` n=1. `filters` : fit multivarié n>1. Sections `Raises:` manquantes (ohlcv, garch, filters).

### 1.C — portfolio : optimiseurs · `[fix/portfolio-optimizers]`
- [ ] **CRITICAL** `ERC` reste bloqué sur 1/N : objectif quartique (~1e-8) < `ftol=1e-6` SLSQP → renvoie le guess initial (contributions au risque non égalisées). API publique stable.
- [ ] **HIGH** `MVP_uc` : même fragilité d'échelle (échoue sur variances faibles) — rescaler l'objectif / `ftol`.
- [ ] **HIGH** entrée mono-actif (N=1) : `np.cov` scalaire → crash IVP/MVP/ERC/HRP. Garder N==1.
- [ ] **HIGH (tests)** tests ERC/MVP_uc tautologiques (variances égales = 1/N) → cas variances inégales + assert contributions au risque / forme fermée MVP.
- [ ] **MEDIUM** `IVP(normalize=True)` double-traite et distord la pondération `1/σ²` ; `_normalize` mute son entrée (copier).
- [ ] **MEDIUM (docs)** `sizing.vol_target` : double-backslash dans la r-string (math non rendue) ; docstrings "Initial weights to maximize" (copier-coller).
- [ ] **MEDIUM (tests)** HRP/MDP : seulement shape/sum → assert comportemental.

### 1.D — metrics / econométrie · `[fix/metrics-econometrics]`
- [ ] **HIGH** `calmar`/`roll_calmar` renvoient `0.0` sur MDD nulle (courbe profitable sans drawdown) alors que `sharpe`/`sortino` renvoient `inf` — convention dénominateur-nul incohérente.
- [ ] **HIGH** `estimator.loglikelihood` mute `h` en place (`h[h==0]=1e-8`) + nom/doc trompeurs (c'est la LL **négative**).
- [ ] **MEDIUM** `econometric_models.ARMA`/`ARMA_GARCH`/`ARMAX_GARCH` crashent sur entrée `list` (MA l'accepte) — `np.asarray`.
- [ ] **MEDIUM** `diversified_ratio` renvoie un array `(1,1)` mais typé `float`.
- [ ] **MEDIUM (docs)** `ARMAX_GARCH` : param `omega` manquant + somme sur régresseurs `Σ_k ψ_k x_{t,k}`.
- [ ] **MEDIUM (tests)** `test_get_parameters` ordres tautologiques (p=q=P=Q=1) ; chemin calmar MDD-nulle non testé.
- [ ] **LOW** `summary` mélange vol `log=True` et sharpe `log=False` ; `target_function` sans docstring ; TeX rendement annualisé + typos.

### 1.E — models : training & losses · `[fix/models-training-loss]`
- [ ] **HIGH** `rolling` (`roll_period > train_period`) : slices à index négatif → wrap sur la queue du tableau = **fuite de futur** dans train/eval. Clamp/garde.
- [ ] **HIGH (docs)** `rolling.run` : `eval_set = slice(t-r,t)` = queue **in-sample** du train, pas OOS — relibeller / corriger.
- [ ] **HIGH** `loss/calmar` : `eps=1e-8` mal dimensionné vs dénominateur (drawdowns O(rendements)) → ratio explose (-3.78e7) et domine le gradient.
- [ ] **MEDIUM** `loss/omega` (et `sortino` plus doux) : même explosion d'échelle eps.
- [ ] **MEDIUM** `rolling._training` : perte train normalisée par `n` et non par nb de batches (insensée selon batch) ; `_display_kpi` lit `loss_eval[-1]` au lieu de `[self.i]`.
- [ ] **MEDIUM** `rolling.run` : fork d'un `multiprocessing.Process` partageant une figure matplotlib (non fork-safe) — rendre en process.
- [ ] **MEDIUM (docs)** `objective.predict` : borne `[-1,1]` vraie seulement pour `position_fn=tanh` par défaut.
- [ ] **LOW** `loss/_base._rf_per_period` mis en cache (muter `rf` = no-op) ; `loss/__init__` omet Calmar/Omega/Hybrid.
- [ ] **LOW (tests)** convention de signe Sortino ; finitude edge calmar/omega/sortino ; reproductibilité refit `ObjectiveModel` ; aucun test du chemin `run()`/`__next__`/`_training` walk-forward.

### 1.F — models : contrat NN · `[fix/models-nn-contract]`
- [ ] **CRITICAL / design** `RNN`/`GRU`/`LSTM` ne récurrent **pas** dans le temps (axe T = batch de `nn.Linear`, aucun état threadé) → implémenter une vraie récurrence séquentielle **ou** redocumenter honnêtement comme couches gated stateless + retirer le contrat `(T,S,N)` impossible.
- [ ] **HIGH** `_base.set_data` ignore `dtype`/`x_type`/`y_type` (float32 demandé → float64) ; float64-input + params float32 crashe.
- [ ] **HIGH** `_base.predict` n'appelle pas `eval()` / `train_on` pas `train()` → dropout fuit en inférence.
- [ ] **MEDIUM** arg `bias` no-op sur RNN/GRU/LSTM ; `predict` récurrent ne respecte pas `predict(X)` du protocole ; `forward_activation=Softmax` par défaut sur cibles de régression.
- [ ] **MEDIUM (tests)** aucun test de causalité RNN/GRU/LSTM ; masquage attention non testé directement.
- [ ] **LOW** `set_seed` bornes + reseed numpy global ; aliasing `from_numpy` ; device `predict` ; typos (`load_model` "Save", type `criterion`) ; pas de doctest mlp/_base/rnn ; asserts loss `>=0` tautologiques + pas de sanity overfit-tiny-batch.

### 1.G — data / core / signal / backtest · `[fix/data-core-signal-backtest]`
- [ ] **HIGH** `signal.rank` : `top+bottom>n_assets` écrase les jambes → non dollar-neutral ; pas de garde négatifs.
- [ ] **HIGH** `data.align.resample` ne marche que sur index `datetime64` (object/int lèvent) + **zéro test**.
- [ ] **MEDIUM** `data.split.walk_forward` : train vide silencieux si `purge>=train` (garde) ; `train_test_split` borne exclusive surprenante (1.0 → 1).
- [ ] **MEDIUM** `core.price_series.to_returns("log"/"pct")` : inf/nan sur prix ≤0 (garde).
- [ ] **MEDIUM** `backtest.engine` (`returns_input=False`) : 1er turnover de position non facturé.
- [ ] **MEDIUM (docs)** `backtest.print_stats` : `underly` traité comme log-returns mais doc dit "prices".
- [ ] **MEDIUM (tests)** `resample`, méthodes `core` (`to_prices`/`apply`/`cumulative`), `_exceptions.ArraySizeError`, branche `value_col` de `frame_to_series`, `strategy.run(X=)` non testés.
- [ ] **LOW** `strategy.run_walk_forward` zéro-te le 1er rendement de chaque bloc test ; doc index `to_prices` ; doc `backtest.loss`/`result` ; doc état initial des mappers.

### 1.H — research : guards & robustesse · `[fix/research]`
- [ ] **HIGH** `ledger.deflated_sharpe` injecte le Sharpe **annualisé** dans `deflated_sharpe_ratio` (qui attend un SR par-période ×√(n-1)) → DSR sature (~0/~1) = confiance fausse. Dé-annualiser.
- [ ] **HIGH** `Ledger` "append-only" mais nom dupliqué **écrase** → `n_trials` sous-compte → corrompt le compte multiple-testing du DSR.
- [ ] **HIGH** `runner.run_experiment` mute `strategy.cost` de l'appelant (jamais restauré) — restaurer en `finally`.
- [ ] **MEDIUM** `leaderboard` trie mal une clé NaN (flotte en tête) ; `Experiment.from_dict` lève sur clé inconnue (pas forward-compatible) ; `load()` laisse échapper `JSONDecodeError`.
- [ ] **MEDIUM** `_seed_everything` utilise `np.random.seed` global (biaise la variance nulle de `permutation_test` pour modèles stochastiques).
- [ ] **MEDIUM (tests)** guards sans assert sur valeur publiée (PSR/DSR Bailey & LdP) ; pas de test corrupt-input / schema-evolution / nom dupliqué.
- [ ] **LOW** `synthetic.regime_switching` démarre toujours en régime 0 ; `_callable_name` renvoie `<lambda>` ; doc `sr_variance=1.0` placeholder + précondition prix>0.

### 1.I — docs : CONTRIBUTING / status / README / Sphinx · `[docs/audit-doc-drift]`
- [ ] **CRITICAL** `CONTRIBUTING.md` : encore `python setup.py build_ext`, "recompile .pyx", `fynance.algorithms`, cadre de déprécation "1.x → 2.0". Tout est pré-2.1 — réécrire pour 2.9 (pure-Python/Numba, `pip install -e ".[dev]"`).
- [ ] **HIGH** `doc/dev/06-status.md` (v2.8.0, comptes 518/593, ATR/ADX/regime/adaptive/market-impact en "Deferred"), `01-overview.md` (v2.8.0 + 518), `05-testing.md` (518) — rafraîchir (v2.9.0 ; ~569 unit + doctests = 658).
- [ ] **MEDIUM** `README.md` : liste de features omet le headline v2.9 (indicateurs OHLCV, feature GARCH causale, fenêtres adaptatives, `RegimeMoE`, `MarketImpactCost`).
- [ ] **MEDIUM/LOW** Sphinx : `iso_vol` (dans `features.__all__`) et l'API registry data (`BaseDataSource`/`register`/`get_source`) non documentés ; `METRICS` dict ; doc/dev/README "NumPy over polars".

## 3. Chore — CI / hygiène

Dette d'outillage repérée pendant le batch library-bricks (v2.9.0). Non bloquant.

- [ ] **CI sur les PR vers `develop`.** `ci.yml` ne s'est pas déclenché sur les PR
  de feature ciblant `develop` (la suite n'a tourné qu'en local) — vérifier les
  triggers du workflow pour que les 4 gates s'exécutent aussi sur ces PR.
- [ ] **Job « Update badges » en échec.** L'étape « Commit badge if changed »
  (workflow `Badges`) échoue à chaque push sur `develop` (problème de push du badge
  de couverture docstring) — corriger les permissions / la condition de commit.

## 2. Research harness — extension optionnelle

Le harnais `fynance.research` est **livré** (S1–S3) : `Experiment`,
`run_experiment`, `write_report`, générateurs synthétiques, garde-fous
(`permutation_test`, `deflated_sharpe_ratio`), `Ledger`/`leaderboard`, multi-input
`X`/`y` et provenance auto-descriptive. Reste optionnel :

- [ ] Explorateur **Streamlit** au-dessus du Ledger (parcourir / filtrer / comparer
  les runs persistés) — interactif, plus tardif.
