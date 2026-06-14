# Plan — §6.1 Optimisations allocation.py

> Plan jetable laissé à la racine (à archiver après). Tâche roadmap §6.1.

## Objectif
Deux items :
1. `HRP()` : remplacer la boucle Python de réordonnancement par du fancy-indexing NumPy.
2. Évaluer un cache de la matrice de covariance entre steps de `rolling_allocation()`.

## Décisions
- **HRP fancy-indexing** : la boucle
  ```python
  w = np.empty(N)
  for i, col_idx in enumerate(sortIx):
      w[col_idx] = w_sorted[i]
  ```
  devient `w = np.empty(N); w[np.asarray(sortIx)] = w_sorted`. Équivalent exact
  (scatter), couvert par les tests HRP existants.
- **Cache covariance** : *évalué, non retenu*. Dans `rolling_allocation`, les
  fenêtres avancent de `s` (par défaut 63) → quasi aucun recouvrement utile ; la
  covariance dépend de toute la fenêtre `[t-n, t]` et changerait à chaque step.
  Un update incrémental O(1) (rank-1) serait complexe et fragile pour un gain
  marginal sur des tailles réalistes. Reporté (noté dans la roadmap).

## Vérification
- `pytest fynance/tests/algorithms/` + suite complète + ruff.
