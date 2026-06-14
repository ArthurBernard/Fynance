# Plan — §5.2 Ensemble direction + magnitude (meta-modèle)

> Plan jetable à la racine (à archiver). Roadmap §5.2 (bloc 🟢).

## Livré (`fynance/models/ensemble.py`)
`StackingEnsemble(direction_factory, magnitude_factory, meta_factory)` :
- modèle direction (ex. `DirectionalAccuracyLoss`) + modèle magnitude (MSE/Sortino).
- `fit_predict(X, y, train/test/roll)` : OOF des 2 bases via `cross_validate`
  (leak-free), empilés comme méta-features, méta-modèle (ex. `SharpeLoss`)
  entraîné dessus. Retourne les prédictions méta (NaN avant le 1er fold).

Anti-fuite : les méta-features sont **out-of-fold** (jamais in-sample des bases).

## Tests (3)
forme (T,M) ; OOF NaN avant 1er fold + fini après ; le méta reçoit 2·M features.

## 🟡 Non fait (besoin de données)
Comparaison empirique vs modèle unique entraîné directement avec SharpeLoss.

## Vérif
- 3 tests ; suite 383 ; ruff + mypy 0.
