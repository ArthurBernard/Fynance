# Plan — §5.5 Normalisation des features

> Plan jetable à la racine (à archiver). Roadmap §5.5 (bloc 🟢).

## État
- **Rolling z-score** : déjà couvert par `roll_standardize` (scale) / `roll_z_score` (stats).
- **Vol-targeting** : déjà couvert par `algorithms.sizing.vol_target` (PR #87).
- **Rank-based normalization** : **ajouté** — `scale.roll_rank` (rang percentile
  glissant strictement passé, [0,1], robuste aux outliers).

## Tests (4)
valeurs (doctest), bornes [0,1], **non-lookahead**, 2D colonne-par-colonne.

## 🟡 Non fait (besoin de données)
Comparaison empirique des 3 approches sur le Sharpe out-of-sample.

## Vérif
- 8 tests scale ; suite 359 ; ruff + mypy 0.
