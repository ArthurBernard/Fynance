# TODO

> Numéros = état de travail. Libres d'être renumérotés quand on
> réorganise. Ils n'apparaissent jamais dans les commits ou le
> CHANGELOG : la traçabilité historique se fait via le numéro de PR.
>
> Une tâche terminée = ligne supprimée (pas de section Done). La trace
> vit dans `CHANGELOG.md` et `git log`.
>
> Workflow : `/pick-task` → plan mode → implémentation → `/finish-task`.
> Release : `/release prepare` quand `[Unreleased]` est suffisamment
> rempli.

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

### 2.1 Compléter les métriques numpy d'évaluation

Dans `fynance/features/metrics.py` (à côté de `sharpe` / `roll_sharpe` existants) :

- [ ] `sortino(X, rf, period, ...)` — symétrique à `sharpe`
- [ ] `directional_accuracy(y_true, y_pred)` — % de signes correctement prédits
- [ ] Tests dans `fynance/tests/features/test_metrics.py` : 6–8 tests

### 2.2 Loss PyTorch différentiables pour l'entraînement

Créer le sous-module `fynance/models/loss/` (un fichier par classe) :

```
fynance/models/loss/
    __init__.py     ← exporte SharpeLoss, SortinoLoss, DirectionalAccuracyLoss
    _base.py        ← BaseLoss : gestion rf, period, eps, TypeError sur numpy input
    sharpe.py       ← SharpeLoss
    sortino.py      ← SortinoLoss
    directional.py  ← DirectionalAccuracyLoss
```

Ops torch pures uniquement — lever `TypeError` si l'input n'est pas un `torch.Tensor`.

- [ ] `SharpeLoss` — `mean(r) / std(r)`, entièrement différentiable
- [ ] `SortinoLoss` — downside via `sqrt(mean(relu(-r)²) + eps)`, différentiable
  (proxy de la downside deviation ; valeur ≠ métrique d'évaluation, documenter)
- [ ] `DirectionalAccuracyLoss` — surrogate `mean(sigmoid(ŷ · y · T))` avec
  température `T` ; différentiable mais proxy (documenter)
- [ ] Paramètres communs dans `BaseLoss` : `rf`, `period` (annualisation), `eps`
- [ ] Tests dans `fynance/tests/models/test_loss.py` : 8–12 tests
  (forward, gradient flow via `loss.backward()`, TypeError sur numpy input, edge cases)

### 2.3 Exemple d'intégration

- [ ] `RollMLP` entraîné avec `SharpeLoss` à la place de MSE (test ou notebook)

## 3. Documentation & notebooks

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

## 6. Optimisations et nettoyage

### 6.1 Supprimer code obsolète dans models/ et backtest/

- [ ] Supprimer `fynance/models/basis.py` — `SignalModel` (stub, `self.y_pred`
  jamais assigné) et `MagnitudeModel` (juste `pass`). Aucune référence ailleurs.
- [ ] Supprimer `class __BacktestNeuralNet` dans `dynamic_plot_backtest.py`
  (marqué "OLD VERSION => DEPRECIATED")
- [ ] Décider du sort de `class _BacktestNeuralNet` (stub TODO, jamais utilisé)

### 6.2 Optimisations allocation.py

- [ ] `HRP()` : remplacer le loop Python de réordonnancement par du fancy indexing
  NumPy (`w[np.array(sortIx)] = w_sorted`)
- [ ] `rolling_allocation()` : nettoyer le double `.bfill()` (lignes ~705, 719)
- [ ] Évaluer cache de la matrice de covariance entre steps consécutifs dans
  `rolling_allocation()` (O(N²) par step, 100 steps × 250 assets = coûteux)

### 6.3 Optimisations rolling.py / _base.py

- [ ] `rolling.py _training()` : supprimer les `print` de debug dans le hot loop
- [ ] `_base.py _set_data()` : utiliser `X.to_numpy(dtype=np.float64, copy=False)`
  pour les DataFrames afin d'éviter les copies accidentelles

### 6.4 Migrer les TODO inline

- [ ] Passer en revue les ~15 commentaires `# TODO` / `# FIXME` dans le code
  (metrics.py, _base.py, allocation.py, _wrappers.py, backtest/) et fermer
  ou migrer vers TODO.md ceux qui restent pertinents
