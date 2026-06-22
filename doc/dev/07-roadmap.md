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
> sont **terminés** — voir `CHANGELOG.md` / `03-decisions.md`. Un **audit complet** (2026-06-22) a été remédié en 9 PR (#188–#196,
> v2.10.0). Un **2ᵉ passage** a relevé une régression (KPI `run()`) + des bugs
> résiduels — backlog en **section 1** ci-dessous ; le reste est optionnel.

> ⚠️ **Hors scope ici** : la recherche de stratégie sur **vraie data** (benchmarks
> empiriques loss / architecture / normalisation, évaluation Sharpe out-of-sample,
> régimes en ligne, sélection de features) vit dans le **repo privé**
> `fynance-research` qui dépend de fynance. La roadmap publique ne tient que du
> code de librairie réutilisable, data-agnostique.

---

## 1. Audit round 2 — régression + bugs résiduels (2026-06-22)

2ᵉ passage d'audit après v2.10.0 : une **régression** introduite par la
remédiation + des bugs **pré-existants** ratés au 1er tour. PR atomiques,
parallélisables (fichiers disjoints). Cible **v2.10.1**.

### 1.1 — `fix/rolling-kpi-crash` (models/rolling.py)
- [ ] **HIGH (régression #196)** `run(backtest_kpi=True)` (chemin par DÉFAUT) lève `IndexError` : `_display_kpi` lit `loss_eval[self.i]` mais `__next__` incrémente `self.i` avant `StopIteration` → après la boucle `self.i==n_iter` (taille `n_iter`). Clamp `min(self.i, n_iter-1)` + test `run(backtest_kpi=True)`.
- [ ] **LOW** `_display_kpi` : dénominateur `T-n-T%s` peut valoir 0 → ZeroDivisionError (garder).

### 1.2 — `fix/features-axis-edges` (features/scale.py, _wrappers.py, features/stats.py)
- [ ] **HIGH (fix #193 incomplet)** `roll_standardize`/`roll_normalize(axis=1)` lèvent encore sur multi-colonnes (params non transposés ; seul `Scale` a été corrigé). Mirror `Scale._apply` + tests standalone axis=1.
- [ ] **MED** `_wrappers.wrap_axis` : `axis=-1` calcule silencieusement `axis=0` — normaliser `axis %= ndim`.
- [ ] **MED** `stats.accuracy`/`directional_accuracy(axis=1)` crashent sur 2-D (seul `y_true` transposé ; transposer les deux).

### 1.3 — `fix/data-core-edges` (data/split.py, data/align.py, core/price_series.py)
- [ ] **HIGH** `split.walk_forward(step<=0)` boucle infinie ; `train_test_split(test_size<0)` → indices train hors-bornes. Gardes.
- [ ] **MED (fix #192 partiel)** `align.resample` ne valide que `kind=='M'` mais polars n'accepte que `[D]/[ms]/[us]/[ns]` ; `[s]`/`[h]`/`[Y]` passent puis erreur opaque. Valider/normaliser la résolution.
- [ ] **MED (fix #192 partiel)** `price_series.to_returns(dropna=False)` et `pnl()` : `IndexError` sur série vide (garde non propagée depuis `apply`).
- [ ] **MED** `align` : index dupliqué silencieusement écrasé (`dict(zip)`) → longueur change. Détecter/lever.
- [ ] **LOW** `from_polars` doc « numeric column » mais filtre par nom seulement.

### 1.4 — `fix/metrics-ratio-sign` (metrics/ratios.py)
- [ ] **MED** `_safe_ratio` renvoie `+inf` pour un excès NÉGATIF sur dénominateur nul (devrait être `-inf`) : `sharpe(flat, rf>0)` = perte sans risque notée meilleure possible. Corriger le signe.
- [ ] **LOW** `roll_sharpe` garde encore la convention `0.0` (incohérent avec `calmar→inf` unifié en #194) — router via `_safe_ratio`.

### 1.5 — `fix/loss-gradient-saturation` (models/loss/{calmar,omega,sortino}.py)
- [ ] **MED (qualité fix #196)** le plancher eps (~5e-9) est trop petit → le clamp `MAX_RATIO=1e3` s'active et ANNULE le gradient en régime faible-risque (Sortino : 78/300 batches). Plancher relatif plus large ou map saturante douce, pour garder un gradient ; doc Calmar honnête.

### 1.6 — `fix/recurrent-fit` (models/_recurrent_base.py, rnn.py, gru.py, lstm.py)
- [ ] **MED** RNN/GRU/LSTM se déclarent `SignalModel` mais `.fit()` lève `TypeError` (train_on exige `H`/`C`). Override `fit`/`predict` (H/C init zéro) OU message clair + retirer la conformité SignalModel.
- [ ] **LOW** `predict` récurrent ne déplace pas X/H/C sur le device du modèle.

### 1.7 — `fix/low-polish` (portfolio/allocation.py, research/{synthetic,compare}.py, features/money_management.py)
- [ ] **LOW** `ERC`/`MVP_uc` : `low_bound>1/N` → poids infaisables ; clamper `low_bound=min(low_bound,1/N)`.
- [ ] **LOW** `_normalize` : `print()` → `warnings.warn` ; test `MVP` pinv via colonne à variance nulle.
- [ ] **LOW** `synthetic.gbm(0)`/`regime_switching(0)` renvoient longueur-1 (garde `n<1`) ; `compare._markdown_table` : union des clés (pas seulement `rows[0]`).
- [ ] **LOW** `money_management.iso_vol` : coercition d'entrée manquante (`np.asarray(...).reshape(-1)`).

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
