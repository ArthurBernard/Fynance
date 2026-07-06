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

## 10. DX one-offs (épic `dx-oneoffs`)

- [ ] `core/checks.py` : `check_conforms(obj, protocol)` + `assert_causal(func)` —
  sonde de lookahead exécutable, dogfoodée dans les tests fynance.
- [ ] Seams DataFrame duck-typées : `to_polars`/`from_pandas` sur `PriceSeries`/
  `OHLCV`/`BacktestResult` (sans dépendance pandas/polars).
