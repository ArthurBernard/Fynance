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
> sont **terminés** — voir `CHANGELOG.md` / `03-decisions.md`. Deux passages d'audit (2026-06-22) ont été **entièrement remédiés** : 9 PR
> (#188–#196, v2.10.0) puis 7 PR (#201–#207, v2.10.1, dont une régression KPI).
> Voir `CHANGELOG.md` / `03-decisions.md`. Il ne reste que l'item optionnel ci-dessous.

> ⚠️ **Hors scope ici** : la recherche de stratégie sur **vraie data** (benchmarks
> empiriques loss / architecture / normalisation, évaluation Sharpe out-of-sample,
> régimes en ligne, sélection de features) vit dans le **repo privé**
> `fynance-research` qui dépend de fynance. La roadmap publique ne tient que du
> code de librairie réutilisable, data-agnostique.

---

## 2. Research harness — extension optionnelle

Le harnais `fynance.research` est **livré** (S1–S3) : `Experiment`,
`run_experiment`, `write_report`, générateurs synthétiques, garde-fous
(`permutation_test`, `deflated_sharpe_ratio`), `Ledger`/`leaderboard`, multi-input
`X`/`y` et provenance auto-descriptive. Reste optionnel :

- [ ] Explorateur **Streamlit** au-dessus du Ledger (parcourir / filtrer / comparer
  les runs persistés) — interactif, plus tardif.

## 3. Harnais R&D multi-actifs / panel (signalé à l'usage par `fynance-research`)

`fynance-research` passe du mono-actif (ETH/BTC) à des stratégies **multi-pair** : un
modèle qui voit `N` paires × `M` features, prédit **signal + magnitude par paire à `H`
horizons**, puis une **règle** (allocation / ranking) décide quoi long/short. Audit du
harnais : le **moteur de backtest gère déjà un book** `(T, N)` (`backtest/engine.py`,
`gross.sum(axis=1)`) et l'**allocation** est livrée (`portfolio.allocation` :
`ERC/HRP/IVP/MVP/MDP` + `rolling_allocation` ; `portfolio.sizing` : `vol_target`,
`kelly_fraction`). **Mais toute la couche R&D au-dessus est câblée mono-actif** — c'est le
blocage. Tâches ordonnées par dépendance ; rétro-compat mono-actif (`N=1`) à préserver.

- [ ] **`ObjectiveModel` — entraînement panel (le cœur de l'enabler).** Aujourd'hui
  `fit(X (T,F), y (T,))`, `out = net(X).reshape(-1)`, `predict → (-1, 1)` : une seule
  colonne de position. Le rendre capable de consommer `X` panel `(T, N, M)` (ou `(T, N·M)`)
  et de sortir un **book de positions** `(T, N)`, avec cible `y` panel `(T, N)` voire
  multi-horizon `(T, N, H)`. Sans ça, aucun modèle cross-actifs n'est entraînable.
- [ ] **Losses book-aware (objectif = portefeuille).** Les `*Loss` (`SharpeLoss`,
  `Sortino`, `Calmar`, …) sont calculées sur un return 1-D. Les rendre capables d'agréger
  le **return du book** `Σ_i pos_i·r_i − coût` avant de scorer, pour qu'on entraîne sur la
  perf du portefeuille et non d'un actif. Ajouter (optionnel) une **loss cross-sectionnelle
  / ranking** différentiable (long top-k vs bottom-k) pour la tête de prédiction. Dépend du
  1er item.
- [ ] **Walk-forward + `run_experiment` multi-actifs.** `Strategy.run` /
  `run_walk_forward(data, y, …)` et `run_experiment` supposent une série de prix unique
  (`n = prices.shape[0]`, une equity, un tearsheet). Accepter `data` **panel** `(T, N)`
  (returns/prix), stitcher un **book OOS**, backtester via le moteur book (déjà OK) et
  renvoyer une **equity de book + attribution par actif**. `run_experiment` : chemin panel
  + provenance (X 3-D, `N` actifs). Dépend du 1er item.
- [ ] **Métrique Information Coefficient / rank-IC + cible multi-horizon.** `fynance.metrics`
  n'a aucune corrélation prédiction↔réalisé. Ajouter `information_coefficient` (Spearman de
  rang, **par horizon**) en évaluation **non-chevauchante** (les labels à `H` barres se
  chevauchent → sinon la skill est surestimée). C'est le **garde-fou predict-then-rule** :
  mesurer la qualité du signal OOS *avant* tout trade, pour distinguer « signal mort »
  (IC≈0) de « signal vivant mangé par les frais / la règle » (IC>0, PnL<0). Optionnel :
  helper `horizon_returns` panel côté `research` / `features`. Indépendant (métrique pure).
- [ ] **Report / tearsheet de book.** Étendre `write_report` + tearsheet (l'axe temporel en
  dates vient d'atterrir, `feat/tearsheet-date-axis`) pour un run multi-actifs : equity
  agrégée + contribution & turnover **par actif**. Dépend du walk-forward panel.

> **Reste côté `fynance-research` (thin, pas ici).** Le réseau bespoke (transformer axial :
> attention temporelle causale + attention cross-actifs) vit dans
> `fynance-research/strategies/nets.py` et ne consomme que le harnais ci-dessus ; les règles
> A/B/C (trend diversifié, cross-sectionnel, stat-arb cointégration) sont des modules
> stratégie. *Si* la brique d'attention cross-sectionnelle s'avère réutilisable et
> data-agnostique, elle pourra remonter en modèle de librairie — à trancher à l'usage.
