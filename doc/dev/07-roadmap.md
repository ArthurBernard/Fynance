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

---


## 5. R&D — Loss, architecture, données

Objectif : identifier empiriquement la meilleure combinaison loss / architecture / features
pour les séries temporelles financières. Chaque sous-tâche est exploratoire :
elle peut déboucher sur du code dans `fynance/models/` ou rester à l'état de
notebook/rapport.

### 5.1 Nouvelles loss functions

Fait (PR #88) : `CalmarLoss`, `OmegaLoss`, `HybridLoss` (α fixe ou learnable).

- [ ] 🟡 **Benchmark empirique** : comparer les loss sur un jeu de données réel
  (out-of-sample Sharpe/Sortino/Calmar/accuracy/drawdown) — nécessite des données.

### 5.2 Architecture : ensemble direction + magnitude avec meta-modèle

- [ ] **Modèle 1 (direction)** : entraîné avec `DirectionalAccuracyLoss`,
  sortie = probabilité de signe positif (sigmoid).
- [ ] **Modèle 2 (magnitude)** : entraîné avec MSE ou `SortinoLoss`,
  sortie = estimation du return.
- [ ] **Meta-modèle** : prend `[signal_1, magnitude_2]` en entrée,
  entraîné avec `SharpeLoss` ou `SortinoLoss`, sur les prédictions out-of-fold
  (walk-forward OOF) pour éviter la fuite de données.
- [ ] Comparer vs. modèle unique entraîné directement avec `SharpeLoss`.

### 5.3 Régimes de marché

- [ ] Identifier des régimes (bull/bear/sideways/high-vol) via HMM ou k-means
  sur les features de volatilité réalisée.
- [ ] Conditionner l'architecture : un modèle par régime (mixture of experts),
  ou un embedding de régime concaténé aux features d'entrée.
- [ ] Évaluer si la détection de régime améliore le Sharpe out-of-sample.

### 5.4 Features / indicateurs techniques

Fait (single-série, causaux) : `roc`, `realized_volatility`, `rolling_skewness`,
`rolling_kurtosis`, `rolling_autocorr` (PR #86) ; déjà présents : EMA/MACD/RSI/
Bollinger/CCI/HMA. Reste **différé** (nécessite une API multi-séries OHLCV) :

- [ ] **ATR / ADX / Williams %R** (High/Low) et **OBV / VWAP** (Volume) — exigent
  des entrées OHLCV ; concevoir une API multi-séries d'abord.
- [ ] **GARCH(1,1) comme feature** (via `fynance.estimator`).

### 5.5 Normalisation des features

Couvert : rolling z-score (`roll_standardize`/`roll_z_score`), vol-targeting
(`sizing.vol_target`), rank-based (`scale.roll_rank`, PR #89).

- [ ] 🟡 Comparer empiriquement les trois approches sur le Sharpe out-of-sample
  (nécessite des données).

### 5.6 Protocole d'entraînement robuste

Fait (PR #90) : purged walk-forward CV (`cross_validate(..., purge=)`),
`exp_sample_weights` (décroissance exponentielle), `EarlyStopping` (sur métrique).
La construction causale des features reste garantie par les property tests
(parité + non-lookahead) — convention continue, pas de code dédié.

- [ ] (optionnel) down-weight basse vol dans `exp_sample_weights` (variante).

### 5.7 R&D rolling — features multi-résolution et efficacité

- [ ] **Multi-résolution** : mêmes features sur plusieurs fenêtres (5/10/21/63) concaténées.
- [ ] **Streaming / mises à jour incrémentales** : prototype `IncrementalFeature`
  (formule récursive O(1) par pas pour GARCH, corrélations).
- [ ] **Fenêtres adaptatives** : `window` variable selon le régime de vol.
- [ ] **Test de causalité de Granger** : filtrer les features sans signal prédictif.

### 5.8 Backtest réaliste

Fait (PR #87) : `algorithms/sizing.py` (`kelly_fraction`, causal `vol_target`,
`transaction_cost` turnover-based) ; métriques `percent_positive`, `tail_ratio`.
Reste optionnel :

- [ ] Slippage / impact de marché non-linéaire au-delà du coût proportionnel.
