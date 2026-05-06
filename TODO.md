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

### 2.1 Pertes financières

Créer `fynance/backtest/loss.py`.

- [ ] `SharpeRatio` loss (annualisée, risk-free rate)
- [ ] `SortinoRatio` loss (downside deviation)
- [ ] `DirectionalAccuracy` loss (% retours correctement prédits)
- [ ] Réduction batch mean, gestion des NaN
- [ ] Tests : 8–12 tests

### 2.2 Intégration PyTorch

- [ ] Wrapper dans `fynance/models/` pour compatibilité `torch.optim`
- [ ] Exemple : `RollMLP` avec `SharpeRatio` loss
- [ ] Tests : 4–6 tests

## 4. Documentation & notebooks

### 4.1 Réécrire les notebooks `Notebooks/`

- [ ] Remplacer exemples Keras par PyTorch
- [ ] Ajouter exemples TCN, Transformer, custom loss functions
- [ ] Montrer rolling walk-forward pour backtest
- [ ] Vérifier que les notebooks tournent (nbval ou papermill)

## 5. Performance & dépendances

### 5.1 Audit des dépendances

- [ ] Audit versions et alternatives modernes des dépendances restantes
  (hors `pandas` → POC `polars` déjà tranché : NumPy natif gagne pour
  l'algèbre matricielle)
