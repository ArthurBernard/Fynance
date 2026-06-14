# Plan — §2 RollMLP entraîné avec SharpeLoss

> Plan jetable à la racine (à archiver). Tâche roadmap §2.

## Objectif
Démontrer (test d'intégration) qu'un `RollMultiLayerPerceptron` s'entraîne avec
`SharpeLoss` (loss financière différentiable) à la place de MSE.

## Mécanisme
- `set_optimizer(SharpeLoss, torch.optim.Adam, lr=...)` → `set_optimizer` fait
  `criterion = SharpeLoss()`.
- Le loop appelle `criterion(outputs, y)` → `SharpeLoss.forward(y_pred=outputs,
  y_true=y)` (y_true ignoré ; le Sharpe ne dépend que des returns prédits).
- Renvoie le Sharpe négatif (scalaire) → `.backward()` met à jour les poids.

## Tests (TestRollMLPWithSharpeLoss)
1. un pas de `_training` tourne, `loss_train[i]` fini.
2. après quelques epochs, au moins un tenseur de poids a bougé (les gradients
   SharpeLoss pilotent bien l'optimiseur).
3. la prédiction garde la bonne forme.

## Vérif
- 3 tests verts ; suite complète + ruff + mypy.
