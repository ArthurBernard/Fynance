# Plan — §5.6 Protocole d'entraînement robuste

> Plan jetable à la racine (à archiver). Roadmap §5.6 (bloc 🟢).

## Livré
- **Purged walk-forward CV** : param `purge` sur `_RollingBasis._fold_slices`
  et `cross_validate` — drop des `purge` dernières obs du train à la bordure
  train/test (Lopez de Prado). Défaut 0 (rétro-compatible).
- **`fynance/models/training.py`** (nouveau) :
  - `exp_sample_weights(n, halflife)` — poids à décroissance exponentielle
    (récent=1, ×½ tous les `halflife`).
  - `EarlyStopping(patience, min_delta, mode)` — arrêt sur métrique (max/min),
    p.ex. maximiser le Sharpe de validation.

## Tests (9)
exp_sample_weights (valeurs/monotonie/halflife invalide) ; EarlyStopping
(patience/reset/mode min/mode invalide) ; purge (`_fold_slices` rétrécit le
train, `cross_validate` tourne).

## Note
« Audit causal de chaque feature » = convention continue, déjà outillée par les
property tests (parité + non-lookahead). Pas de code dédié.

## Vérif
- 9 tests ; suite 369 ; ruff + mypy 0.
