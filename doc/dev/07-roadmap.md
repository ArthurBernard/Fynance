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

> **fynance 2.x livré** (v2.8.0 sur `master`/PyPI). Le refactor en couches, le port
> Cython→Numba (build pure-Python), le harnais R&D `fynance.research` (S1–S3) et la
> brique d'entraînement aligné-objectif (`ObjectiveModel` : objectif différentiable,
> net de coûts, mini-batch) sont **terminés** — voir `CHANGELOG.md` / `03-decisions.md`.
> Reste, ci-dessous, des **bricks de librairie** non bloquées par la donnée.

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

## 1. Bricks library (non bloquées par la donnée)

Travail de librairie pur, testable sur data synthétique. Exploratoire : peut
déboucher sur du code dans `fynance/` ou rester à l'état de rapport.

### 1.1 Indicateurs OHLCV multi-séries

Single-série & causaux déjà présents : EMA/MACD/RSI/Bollinger/CCI/HMA, `roc`,
`realized_volatility`, `rolling_skewness`/`kurtosis`/`autocorr`. Manquent les
indicateurs exigeant plusieurs séries :

- [ ] Concevoir une **API multi-séries OHLCV** d'abord, puis **ATR / ADX /
  Williams %R** (High/Low) et **OBV / VWAP** (Volume).
- [ ] **GARCH(1,1) comme feature** (via `fynance.estimator`).

### 1.2 Architecture conditionnée par le régime

`RegimeDetector` causal (fit-on-train / assign-online) est déjà livré ; reste à
s'en servir pour piloter l'architecture :

- [ ] Conditionner le modèle sur le régime — **mixture-of-experts** /
  embedding de régime.

### 1.3 Fenêtres adaptatives (dépend de §1.2)

- [ ] `window` variable selon le régime de volatilité.

### 1.4 Backtest réaliste

Fait : `portfolio/sizing.py` (`kelly_fraction`, causal `vol_target`,
`transaction_cost` turnover-based) ; métriques `percent_positive`, `tail_ratio`.
Reste optionnel :

- [ ] **Slippage / impact de marché non-linéaire** au-delà du coût proportionnel.
