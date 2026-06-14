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

## 1. Nouvelles architectures ML

### 1.2 Transformer financier

Compléter `fynance/models/transformer.py`.

- [ ] Positional encoding pour séries temporelles (relatif vs absolu)
- [ ] Masking causal (pas de lookahead)
- [ ] Tester multi-head attention avec données réelles
- [ ] Tests : 6–10 tests

## 3. Documentation & notebooks

### 3.1 URL de la documentation

- [ ] Évaluer remplacement de GitHub Pages par RTD (URL `fynance.readthedocs.io`)
  via trigger manuel de build RTD depuis le workflow GitHub Actions à chaque merge
  sur `master` (API RTD webhook), ou custom domain sur GitHub Pages.

### 3.2 Réécrire les notebooks `Notebooks/`

- [ ] Remplacer exemples Keras par PyTorch
- [ ] Ajouter exemples TCN, Transformer, custom loss functions
- [ ] Montrer rolling walk-forward pour backtest
- [ ] Vérifier que les notebooks tournent (nbval ou papermill)

## 4. Performance & dépendances

- [ ] Audit versions et alternatives modernes des dépendances restantes.
  (pandas a été remplacé par polars en entrée + numpy en sortie ; le cœur
  algébrique reste NumPy natif.)

## 5. R&D — Loss, architecture, données

Objectif : identifier empiriquement la meilleure combinaison loss / architecture / features
pour les séries temporelles financières. Chaque sous-tâche est exploratoire :
elle peut déboucher sur du code dans `fynance/models/` ou rester à l'état de
notebook/rapport.

### 5.1 Nouvelles loss functions

- [ ] **Calmar loss** — proxy différentiable du max drawdown
  (`max_drawdown ≈ max(cumsum(-r)) − min(cumsum(-r))` via `torch.cumsum` + `torch.max`).
  Valeur à annualiser pour comparer avec la métrique d'évaluation.
- [ ] **Omega loss** — ratio gains/pertes pondérés au-delà d'un seuil `L` :
  `Ω = E[max(r−L, 0)] / E[max(L−r, 0) + ε]`, entièrement différentiable avec `F.relu`.
- [ ] **Loss hybride multi-objectif** — combiner deux objectifs avec un poids `α` :
  `L = α·SharpeLoss + (1−α)·DirectionalAccuracyLoss`. Chercher `α` optimal
  par grid-search ou en le rendant learnable (paramètre `nn.Parameter`).
- [ ] **Benchmark empirique** : comparer les loss sur un jeu de données réel
  (même walk-forward, mêmes features) — métriques out-of-sample : Sharpe, Sortino,
  Calmar, accuracy directionnelle, drawdown max.

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

- [ ] **Tendance** : EMA, MACD, ADX.
- [ ] **Momentum** : RSI, ROC, Williams %R.
- [ ] **Volatilité** : ATR, Bollinger Band width, vol réalisée glissante.
- [ ] **Volume** : OBV, VWAP (si données intraday disponibles).
- [ ] **Statistiques glissantes** : autocorrélation, skewness, kurtosis
  des returns sur fenêtres multiples (5, 10, 21, 63 jours).
- [ ] **Vol conditionnelle** : GARCH(1,1) comme feature (via `fynance.estimator`).
- [ ] Implémenter dans `fynance/features/` avec décorateurs `@WrapperArray` existants.

### 5.5 Normalisation des features

- [ ] **Rolling z-score** : `(x − μ_t) / σ_t` avec fenêtre strictement passée.
- [ ] **Rank-based normalization** : rangs percentiles sur la fenêtre passée.
- [ ] **Vol-targeting** : normaliser les returns par la vol réalisée passée.
- [ ] Comparer empiriquement les trois approches sur le Sharpe out-of-sample.

### 5.6 Protocole d'entraînement robuste

Construction causale des features (protection primaire contre la fuite) : toute
feature à `t` doit s'écrire `f(data[t-window:t])`. Garantie côté kernels par les
tests de propriété (parité + non-lookahead) ; à étendre aux nouvelles features.

- [ ] **Audit causal de chaque feature** (pas de `rolling(center=True)`, pas de
  normalisation globale, pas de z-score incluant la fenêtre test).
- [ ] **Purged walk-forward CV** : exclure la zone de bordure train/test quand le
  nombre de folds est grand ou la fenêtre feature >> horizon.
- [ ] **Pondération des échantillons** : decay exponentiel / down-weight basse vol.
- [ ] **Early stopping sur métrique financière** (maximiser le Sharpe de validation).

### 5.7 R&D rolling — features multi-résolution et efficacité

- [ ] **Multi-résolution** : mêmes features sur plusieurs fenêtres (5/10/21/63) concaténées.
- [ ] **Streaming / mises à jour incrémentales** : prototype `IncrementalFeature`
  (formule récursive O(1) par pas pour GARCH, corrélations).
- [ ] **Fenêtres adaptatives** : `window` variable selon le régime de vol.
- [ ] **Test de causalité de Granger** : filtrer les features sans signal prédictif.

### 5.8 Backtest réaliste

- [ ] **Coûts de transaction** : spread bid-ask + slippage dans le P&L (BPS/trade, impact linéaire).
- [ ] **Position sizing** : Kelly fractionnel + vol-targeting dans `fynance/algorithms/`.
- [ ] **Métriques de robustesse** : tail ratio, % de mois positifs, etc. — compléter `fynance/features/`.
