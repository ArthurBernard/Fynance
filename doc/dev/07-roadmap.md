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

> **fynance 2.x livré** (v2.10.2 sur `master`/PyPI). Le refactor en couches, le port
> Cython→Numba (build pure-Python), le harnais R&D `fynance.research` (S1–S3), la
> brique d'entraînement aligné-objectif (`ObjectiveModel`) et les **bricks de
> librairie** (conteneur `OHLCV` + indicateurs ATR/ADX/Williams %R/OBV/VWAP, feature
> GARCH causale, `RegimeMoE`, fenêtres adaptatives, coût market-impact non-linéaire)
> sont **terminés** — voir `CHANGELOG.md` / `03-decisions.md`. Deux passages d'audit (2026-06-22) ont été **entièrement remédiés** : 9 PR
> (#188–#196, v2.10.0) puis 7 PR (#201–#207, v2.10.1, dont une régression KPI).
> Voir `CHANGELOG.md` / `03-decisions.md`. L'épic **harnais R&D multi-actifs /
> panel** (2026-06-23) est **livré** : 5 PR (#215–#219) — `ObjectiveModel` panel,
> losses book-aware + `RankingLoss`, `information_coefficient`/`horizon_returns`,
> walk-forward + attribution par actif, report de book. Le backlog §3–§10 vient
> du **workflow d'idéation 2026-07** (43 propositions générées sous 6 angles,
> scorées valeur/fit/faisabilité par panel de juges, 29 retenues) — détail
> complet (descriptions, API sketches, scores) dans
> `plans/_catalog/feature-catalog-2026-07.md` (local).

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

## 3. Multi-actifs non-crypto — calendrier & coûts (prérequis d'honnêteté)

Déposé 2026-07-04 par fynance-research : sa prochaine campagne (book trend /
cross-section sur un univers ETF, données via dccd) est **bloquée** par deux
hypothèses crypto câblées dans le harnais. Code de librairie data-agnostique —
c'est bien ici que ça vit, la campagne elle-même reste dans le repo privé.

> **MàJ 2026-07-06 (v2.13.0) — primitives livrées, reste le câblage.** L'épic
> `backtest-realism` a livré les briques data-agnostiques de ces deux items ;
> ce qui reste est leur intégration dans le chemin walk-forward.

- [ ] **Calendrier de sessions.** _Primitives livrées_ : `fynance.data.sessions`
  (`session_mask`/`session_id`/`session_bounds`/`split_sessions`) tague et
  découpe les sessions sur des timestamps epoch. _Reste_ : les câbler dans
  `walk_forward` (coupure train/test sur frontières de séance), l'agrégation
  intra-session, et traiter le **gap overnight** comme un rendement à part (pas
  un pas intraday) — sans régresser le chemin crypto 24/7 par défaut. NB : DST
  est hors scope de `data.sessions` (offset fixe) — à trancher ici.
- [ ] **Modèle de coûts actions.** _Primitives livrées_ : `HoldingCost`
  (commission implicite via `ProportionalCost`, **borrow** du short, financement
  du levier, crédit de cash) + `CompositeCost` pour empiler. _Reste_ : le
  brancher par **configuration** (pas par fork) sur le chemin de recherche
  actions, en remplacement du couple taker-fee + funding-perp crypto.
- [ ] **Coût du turnover implicite d'un book qui drift.** Remonté par les agents
  pendant l'épic `backtest-realism` : le coût turnover du moteur `backtest()`
  **sous-compte** le trading réel d'un book dont les poids dérivent avec les
  rendements entre deux rebalancements — il mesure `Σ|E_t − E_{t-1}|` (equity)
  au lieu du trade effectif `Σ|w_t − drift(w_{t-1}, r_t)|`. Les briques
  `portfolio.rebalance` produisent déjà le book effectif correct ; il reste à
  faire refléter ce coût dans les KPIs du moteur (petit chantier moteur isolé,
  API `backtest()` stable à préserver).

## 4. Crypto-perp & guardrails de recherche (déposé 2026-07-06 par fynance-research)

Trois briques data-agnostiques identifiées par la campagne allweather-wave (E59–E74,
233 trials — voir le journal du repo privé). Chacune corrige un biais *mesuré* du
harnais, pas une envie spéculative.

- [ ] **Coût de funding perp signé par jambe.** Le standard de recherche actuel
  applique un forfait a-priori (11 %/an sur le notionnel long) qui **mal-signe les
  books de carry** : un book long des coins à funding négatif / short des coins à
  funding positif *reçoit* du funding sur ses deux jambes (constaté E59/E67/E69 —
  le forfait retranche ~0.2 de Sharpe à un book qui devrait être crédité). La série
  réelle est désormais dans le store dccd
  (`<exchange>/funding/<PAIR>_PERP/<year>.parquet`, colonnes `TS/rate/mark_price`,
  settlements 8 h, Binance/Bybit/krakenfutures, 2019+) : il manque la brique de coût
  qui crédite/débite `position × rate` à chaque settlement **par signe de jambe** —
  un `FundingCost(rate_series)` composable via `CompositeCost` (primitives v2.13.0
  déjà livrées), avec fallback taux-constant quand aucune série n'est fournie.
  Remplace le forfait sur le chemin perp.
- [ ] **Nulls de permutation corrects pour books de position.** Deux pièges
  documentés par la campagne : (a) réordonner les *returns nets* d'un book est
  Sharpe-invariant — le p=1.0 est mécanique (E67) ; le null correct **désaligne
  positions ↔ forward-returns**. (b) Sur un book long-biaisé d'actifs à fort drift,
  le bootstrap par blocs fabrique des chemins synthétiques *plus* trendy que le réel
  (perm mean 1.52 > réel 1.07 → p ininterprétable, E62) ; il faut une variante
  **dé-driftée / sign-flip**, ou le benchmark always-long comme null explicite.
  À loger à côté de `block_permutation_test` dans `fynance.research`.
- [ ] **IC outillé : t HAC/Newey-West, Fama-MacBeth, IC incrémental.** Chaque agent
  de screen de la campagne a re-dérivé le même trio : t de Newey-West sur la série
  d'IC quotidiens (fenêtres qui se recouvrent), IC panel Fama-MacBeth, et l'**IC
  incrémental** après résidualisation OLS sur un signal de contrôle (la barre
  « pas une copie du régime prix » qui a tué E57/E65-échos/E72). Promouvoir en
  utilitaires à côté d'`information_coefficient`/`ic_summary` dans
  `fynance.metrics` pour arrêter la re-dérivation feuille à feuille.

## 5. Outillage / CI (maintenance)

- [ ] **Actions GitHub sur Node.js 20 déprécié.** Le run `release.yml` avertit
  que `actions/checkout@v4` et `softprops/action-gh-release@v2` ciblent Node 20
  (forcé sur Node 24 par les runners pour l'instant, cassera à terme). Bumper
  ces actions vers leurs versions Node 24 quand elles sortent (surveiller aussi
  les workflows `ci.yml`). Repéré au tag v2.13.1 (2026-07-06) — sans impact
  immédiat, purement préventif.
