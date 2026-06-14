# Plan — §5.7 R&D rolling : multi-résolution & efficacité

> Plan jetable à la racine (à archiver). Roadmap §5.7 (bloc 🟢).

## Livré (`fynance/features/engineering.py`)
- `multi_resolution(func, X, windows)` — empile une feature window-based à
  plusieurs résolutions (colonne par fenêtre).
- `granger_causality(x, y, lag)` — test F de causalité de Granger (filtrage de
  features) → (F, p-value) via OLS numpy + scipy.f.
- `IncrementalMoments` — moyenne/variance en ligne O(1) (Welford), prototype
  pour mises à jour incrémentales.

## Tests (6)
multi_res (forme + colonnes) ; granger (cause détectée p<0.01, indépendant
p>0.05, série trop courte lève) ; IncrementalMoments (== numpy batch, chaînage).

## Différé
**Fenêtres adaptatives** (window selon régime de vol) — flou, dépend de §5.3
(détection de régime). À traiter après 5.3.

## Vérif
- 6 tests ; suite 377 ; ruff + mypy 0 ; doctests (multi_resolution, IncrementalMoments).
