# Plan — §5.1 Nouvelles loss functions

> Plan jetable à la racine (à archiver). Roadmap §5.1 (bloc 🟢).

## Livré (dans `fynance/models/loss/`, calqués sur SharpeLoss)
- `CalmarLoss` — Calmar négatif ; max drawdown différentiable via `torch.cummax`.
- `OmegaLoss` — Omega négatif `E[relu(r-L)] / E[relu(L-r)]` (seuil `threshold`).
- `HybridLoss` — combinaison convexe `α·L_a + (1-α)·L_b` ; `α` fixe **ou
  learnable** (`nn.Parameter` via sigmoid). Forwarde `y_true` aux composants
  (ex. DirectionalAccuracyLoss qui en a besoin).

Exportés via `fynance.models.loss` et `fynance.models`.

## Tests (9 + intégration)
Calmar (scalaire+grad, rejet non-tensor), Omega (valeur connue, seuil+grad),
Hybrid (somme pondérée, forward y_true, α learnable optimisé), entraînement
d'un MLP avec CalmarLoss.

## 🟡 Non fait (besoin de données)
« Benchmark empirique » des loss sur dataset réel (out-of-sample Sharpe/Sortino/…).

## Vérif
- 26 tests loss (dont 9 neufs) ; suite 354 ; ruff + mypy 0.
