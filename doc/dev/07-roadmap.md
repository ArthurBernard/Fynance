# 7 — Roadmap / next steps

This file is the **single source of truth** for open work — read by `/pick-task`,
updated by `/finish-task` / `/abandon-task`. **Local / gitignored** (R&D stays
private). Finished work is *removed* from here: git log + `CHANGELOG.md` are
authoritative for *what* shipped, `03-decisions.md` for *why*. Keep it short and
true.

> Les numéros de section = état de travail, librement renumérotables. Ils
> n'apparaissent jamais dans les commits/CHANGELOG — la traçabilité passe par le
> numéro de PR. Tâche terminée = ligne supprimée (pas de section Done).
>
> Loop : `/pick-task` → `/plan` (arbre dans `plans/`) → `/execute-leaf` →
> `/finish-task`. Release : `/release` quand `[Unreleased]` est suffisamment rempli.

---

## 1. Nouvelles architectures ML

### 1.1 Temporal Convolutional Network (TCN)

Implémenter dans `fynance/models/tcn.py`.

- [ ] Base `BaseNeuralNet` avec walk-forward iterator
- [ ] Couches dilated 1D convolutions + residual blocks
- [ ] Architecture similaire à `recurrent_neural_network.py` (MLP, GRU, LSTM)
- [ ] Tests : 5–8 tests (forward pass, gradient flow, rolling, edge cases)

### 1.2 Transformer financier

Compléter `fynance/models/transformer.py`.

- [ ] Positional encoding pour séries temporelles (relatif vs absolu)
- [ ] Masking causal (pas de lookahead)
- [ ] Tester multi-head attention avec données réelles
- [ ] Tests : 6–10 tests

## 2. Loss functions custom pour optimisation

Architecture retenue (Option C) : formules numpy dans `fynance/features/metrics.py`
(évaluation/backtest), réimplémentées en ops torch pures dans `fynance/models/loss.py`
(entraînement). Les deux paths sont indépendantes — pas de conversion numpy↔torch.

### 2.3 Exemple d'intégration

- [ ] `RollMLP` entraîné avec `SharpeLoss` à la place de MSE (test ou notebook)

## 3. Documentation & notebooks

### 3.0 URL de la documentation

- [ ] Évaluer remplacement de GitHub Pages par RTD (URL `fynance.readthedocs.io`)
  via trigger manuel de build RTD depuis le workflow GitHub Actions à chaque merge
  sur `master` (API RTD webhook), ou custom domain sur GitHub Pages.

### 3.1 Réécrire les notebooks `Notebooks/`

- [ ] Remplacer exemples Keras par PyTorch
- [ ] Ajouter exemples TCN, Transformer, custom loss functions
- [ ] Montrer rolling walk-forward pour backtest
- [ ] Vérifier que les notebooks tournent (nbval ou papermill)

## 4. Performance & dépendances

### 4.1 Audit des dépendances

- [ ] Audit versions et alternatives modernes des dépendances restantes
  (hors `pandas` → POC `polars` déjà tranché : NumPy natif gagne pour
  l'algèbre matricielle)

## 5. Refactoring structurel

### 5.1 Éclater `fynance/features/metrics.py` (1 800 lignes)

Découper en 4 modules thématiques dans `fynance/features/` (pas de façade
`metrics.py` : `features/__init__.py` importe directement depuis les nouveaux
fichiers). Les helpers privés partagés vont dans `_metrics_helpers.py`.

Découpage proposé :

```
fynance/features/
    _metrics_helpers.py   ← _annual_return, _annual_volatility,
                             _annual_downside_volatility,
                             _roll_annual_return, _roll_annual_volatility,
                             _drawdown, _roll_drawdown, _roll_mdd
    returns.py            ← annual_return, annual_volatility,
                             roll_annual_return, roll_annual_volatility,
                             perf_index, perf_returns
    ratios.py             ← sharpe, sortino, calmar, diversified_ratio,
                             roll_sharpe, roll_calmar
    drawdown.py           ← drawdown, mdd, roll_drawdown, roll_mdd
    stats.py              ← z_score, accuracy, directional_accuracy,
                             mad, roll_mad, roll_z_score
```

`metrics_cy.pyx` reste intact (Cython, extend only).
Mettre à jour `features/__init__.py` pour importer depuis les 4 nouveaux
fichiers (+ `metrics_cy`). Supprimer `metrics.py`.

- [ ] Créer `_metrics_helpers.py` avec tous les helpers privés
- [ ] Créer `returns.py`, `ratios.py`, `drawdown.py`, `stats.py`
- [ ] Mettre à jour `features/__init__.py` (supprimer import `metrics`)
- [ ] Supprimer `metrics.py`
- [ ] Vérifier que tous les tests et doctests passent (`pytest` + `--doctest-modules`)

### 5.2 Éclater les fichiers multi-classes de `fynance/models/`

Candidats identifiés :

- `attention.py` → `scaled_dot_product_attention.py` + `multi_head_attention.py`
- `econometric_models.py` → un fichier par modèle : `ma.py`, `arma.py`,
  `arma_garch.py`, `armax_garch.py` + `get_parameters.py`
- `rolling.py` → `_rolling_basis.py` + `roll_mlp.py` (CVResult reste dans
  `roll_mlp.py`, très couplé à `RollMultiLayerPerceptron`)
- `lstm.py` et `gru.py` — les hiérarchies internes (`_LSTMCell → LSTMCell →
  LongShortTermMemory`) sont trop couplées pour être séparées : laisser en l'état

- [ ] Éclater `attention.py`
- [ ] Éclater `econometric_models.py`
- [ ] Éclater `rolling.py`
- [ ] Mettre à jour `models/__init__.py` pour chaque éclatement

### 5.3 Éclater `fynance/backtest/dynamic_plot_backtest.py` (708 lignes, 6 classes)

Candidats : `DynaPlotBackTest`, `DynaPlotAccuracy`, `DynaPlotLoss`,
`DynaPlotPerf`, `BacktestNeuralNet` (+ `_BacktestNeuralNet` privé).

- [ ] Un fichier par classe publique substantielle
- [ ] Mettre à jour `backtest/__init__.py`

## 7. R&D — Loss, architecture, données

Objectif : identifier empiriquement la meilleure combinaison loss / architecture / features
pour les séries temporelles financières. Chaque sous-tâche est exploratoire :
elle peut déboucher sur du code dans `fynance/models/` ou rester à l'état de
notebook/rapport.

### 7.1 Nouvelles loss functions

- [ ] **Calmar loss** — proxy différentiable du max drawdown
  (`max_drawdown ≈ max(cumsum(-r)) − min(cumsum(-r))` via `torch.cumsum` + `torch.max`).
  Valeur à annualiser pour comparer avec la métrique d'évaluation.
- [ ] **Omega loss** — ratio gains/pertes pondérés au-delà d'un seuil `L` :
  `Ω = E[max(r−L, 0)] / E[max(L−r, 0) + ε]`, entièrement différentiable avec `F.relu`.
- [ ] **Loss hybride multi-objectif** — combiner deux objectifs avec un poids `α` :
  `L = α·SharpeLoss + (1−α)·DirectionalAccuracyLoss`. Chercher `α` optimal
  par grid-search ou en le rendant learnable (paramètre `nn.Parameter`).
- [ ] **Benchmark empirique** : comparer les 5 loss sur un jeu de données réel
  (même walk-forward, mêmes features) — métriques out-of-sample : Sharpe, Sortino,
  Calmar, accuracy directionnelle, drawdown max.

### 7.2 Architecture : ensemble direction + magnitude avec meta-modèle

Suite de la discussion sur le stacking à trois niveaux :

- [ ] **Modèle 1 (direction)** : entraîné avec `DirectionalAccuracyLoss`,
  sortie = probabilité de signe positif (sigmoid).
- [ ] **Modèle 2 (magnitude)** : entraîné avec MSE ou `SortinoLoss`,
  sortie = estimation du return.
- [ ] **Meta-modèle** : prend `[signal_1, magnitude_2]` en entrée,
  entraîné avec `SharpeLoss` ou `SortinoLoss`.
  Entraîner sur les prédictions out-of-fold (walk-forward OOF) pour éviter
  la fuite de données — jamais sur les mêmes fenêtres que les sous-modèles.
- [ ] Comparer vs. modèle unique entraîné directement avec `SharpeLoss`.

### 7.3 Régimes de marché

- [ ] Identifier des régimes (bull/bear/sideways/high-vol) via HMM ou k-means
  sur les features de volatilité réalisée.
- [ ] Conditionner l'architecture : soit un modèle par régime (mixture of experts),
  soit un embedding de régime concaténé aux features d'entrée.
- [ ] Évaluer si la détection de régime améliore le Sharpe out-of-sample.

### 7.4 Features / indicateurs techniques

- [ ] **Indicateurs de tendance** : EMA, MACD, ADX.
- [ ] **Indicateurs de momentum** : RSI, ROC, Williams %R.
- [ ] **Indicateurs de volatilité** : ATR, Bollinger Band width, vol réalisée glissante.
- [ ] **Indicateurs de volume** : OBV, VWAP (si données intraday disponibles).
- [ ] **Features statistiques glissantes** : autocorrélation, skewness, kurtosis
  des returns sur fenêtres multiples (5, 10, 21, 63 jours).
- [ ] **Vol conditionnelle** : GARCH(1,1) comme feature de régime de vol
  (utiliser `fynance.estimator`).
- [ ] Implémenter dans `fynance/features/` avec décorateurs `@WrapperArray` existants.

### 7.5 Normalisation des features

- [ ] **Rolling z-score** : `(x − μ_t) / σ_t` avec fenêtre glissante strictement
  passée — évite toute fuite de données.
- [ ] **Rank-based normalization** : transformer les features en rangs percentiles
  sur la fenêtre passée — robuste aux outliers.
- [ ] **Vol-targeting** : normaliser les returns en entrée par la vol réalisée
  passée (target = vol annuelle constante, ex. 15 %).
- [ ] Comparer empiriquement les trois approches sur le Sharpe out-of-sample.

### 7.6 Protocole d'entraînement robuste

**Construction causale des features** (protection primaire contre la fuite) :
toute feature à `t` doit s'écrire `f(data[t-window:t])` — jamais de données futures.
C'est ce que font déjà `roll_sharpe`, `roll_mdd`, etc. ; le vérifier systématiquement
pour les nouveaux indicateurs de 7.4 et 7.5.

- [ ] **Audit causal de chaque feature** : pour chaque indicateur implémenté,
  vérifier que le calcul est strictement passé (pas de `df.rolling(window, center=True)`,
  pas de normalisation globale sur tout le dataset, pas de z-score calculé sur la
  fenêtre test incluse).
- [ ] **Purged walk-forward CV** : dans un contexte multi-fold (k-fold temporel),
  les features adjacentes au bord train/test partagent `window-1` jours de données
  sous-jacentes → les premières `window` observations du fold test sont corrélées
  avec les dernières du fold train. Exclure cette zone de bordure (purging) quand
  le nombre de folds est grand ou que la fenêtre feature >> horizon de prédiction.
  En walk-forward linéaire classique (un seul train → test avançant), ce problème
  est moins critique si les features sont causales.
- [ ] **Pondération des échantillons** : down-weighter les observations anciennes
  (decay exponentiel) ou celles en régime de faible vol (signal moins informatif).
- [ ] **Early stopping sur métrique financière** : arrêter l'entraînement sur
  la validation en maximisant le Sharpe plutôt qu'en minimisant la loss.

### 7.8 R&D rolling — features multi-résolution et efficacité

- [ ] **Multi-résolution** : construire les mêmes features (vol, momentum, z-score)
  sur plusieurs fenêtres simultanément (5, 10, 21, 63 jours) et concaténer —
  laisser le modèle apprendre l'horizon pertinent plutôt que de le fixer.
- [ ] **Streaming / mises à jour incrémentales** : évaluer si les features les plus
  coûteuses (GARCH, corrélations croisées) peuvent être mises à jour en O(1) par
  pas de temps plutôt que recalculées en O(window). Implémenter un prototype
  `IncrementalFeature` basé sur une formule récursive.
- [ ] **Fenêtres adaptatives** : faire varier `window` en fonction du régime de
  vol (courte en haute vol, longue en basse vol) pour que les features capturent
  une quantité d'information constante plutôt qu'une durée fixe.
- [ ] **Test de causalité de Granger** : pour chaque feature candidate, mesurer
  empiriquement si elle prédit le return à t+1 au-delà de l'autocorrélation des
  returns — filtrer les features sans signal avant d'entraîner.

### 7.7 Backtest réaliste

- [ ] **Coûts de transaction** : intégrer spread bid-ask + slippage dans le calcul
  du P&L de backtest (paramétrable : BPS par trade, impact de marché linéaire).
- [ ] **Position sizing** : implémenter Kelly fractionnel et vol-targeting comme
  modules de `fynance/algorithms/` ; tester leur impact sur le Sharpe net.
- [ ] **Métriques de robustesse** : taux de calmar, Omega ratio, tail ratio,
  % de mois positifs — compléter `fynance/features/metrics.py`.

## 6. Optimisations et nettoyage

### 6.1 Supprimer code obsolète dans models/ et backtest/

- [x] Supprimer `fynance/models/basis.py` — `SignalModel` (stub, `self.y_pred`
  jamais assigné) et `MagnitudeModel` (juste `pass`). ✅ PR #58.
- [x] Supprimer `class __BacktestNeuralNet` dans `dynamic_plot_backtest.py`
  (marqué "OLD VERSION => DEPRECIATED"). ✅ PR #58.
- [x] Décider du sort de `class _BacktestNeuralNet` (stub TODO) — supprimé. ✅ PR #58.

### 6.2 Optimisations allocation.py

- [ ] `HRP()` : remplacer le loop Python de réordonnancement par du fancy indexing
  NumPy (`w[np.array(sortIx)] = w_sorted`)
- [x] `rolling_allocation()` : réécrit pandas-free en numpy (le double `.bfill()` est désormais un seul helper `_bfill`). ✅ PR #61.
- [ ] Évaluer cache de la matrice de covariance entre steps consécutifs dans
  `rolling_allocation()` (O(N²) par step, 100 steps × 250 assets = coûteux)

### 6.3 Optimisations rolling.py / _base.py

- [x] `rolling.py _training()` : `print` de debug supprimés. ✅ PR #58.
- [x] `_base.py _set_data()` : entrée pandas remplacée par polars (`X.to_numpy()`).
  ✅ PR #61. (Reste optionnel : passer `dtype=np.float64` à `to_numpy`.)

### 6.4 Migrer les TODO inline

- [ ] Passer en revue les ~15 commentaires `# TODO` / `# FIXME` dans le code
  (metrics.py, _base.py, allocation.py, _wrappers.py, backtest/) et fermer
  ou migrer vers TODO.md ceux qui restent pertinents


## 8. Qualité & dette technique (audit 2026-06-14)

Items issus de l'audit complet du dépôt **non couverts ailleurs** dans cette
roadmap. Le reste de l'audit a déjà été traité (PRs #58–#62) ou figure dans les
sections 3, 5 et 6 ci-dessus.

### 8.1 `estimator/estimator.py` — décider du sort
- [x] `estimation()` est annoncée « NOT YET WORKING ! » / « NEED TO FIND AN
  OPTIMIZER » mais reste exposée. → lève désormais `NotImplementedError` +
  docstring « experimental », pointe vers `get_parameters`. ✅ PR #63.

### 8.2 Typage : résorber les 104 erreurs mypy puis ajouter le gate CI
- [ ] `mypy fynance/` = 104 erreurs (bruit numpy-2 `Returning Any`, hiérarchies
  torch `predict`/`train_on` incompatibles, `print_stats.py` variable `bool`
  réassignée en `ndarray`). Les corriger par lots.
- [ ] Une fois à 0, ajouter un job `mypy` dans `ci.yml` (à côté de
  ruff / interrogate / docs).

### 8.3 Couverture du cœur causal
- [ ] `models/rolling.py` (64 %) — couvrir la boucle d'entraînement `_training`.
- [ ] `algorithms/rolling.py` (22 %) — walk-forward allocation.
- [ ] `models/econometric_models.py` (54 %) — chemins ARMA/GARCH.
- [ ] `algorithms/browsers.py` (0 %) — clarifier le rôle puis tester ou retirer.

### 8.4 Tests de propriété (transformer des conventions en garde-fous)
- [ ] **Parité py ↔ cy** : suite paramétrée
  `assert np.allclose(py_impl(x), cy_impl(x))` pour chaque paire
  `metrics` / `momentums` / `roll_functions` (aurait attrapé le faux-FIXME
  `momentums_cy.pyx:26`).
- [ ] **Non-lookahead générique** : perturber `X[t+1:]` et vérifier que `f(X)[t]`
  est inchangé, pour chaque feature rolling.

### 8.5 Compléter les inventaires de doc/dev
- [ ] Ajouter `features/money_management.py`, `_exceptions.py` et le sous-package
  `models/loss/` aux inventaires de `01-overview.md` / `04-subpackages.md`.
