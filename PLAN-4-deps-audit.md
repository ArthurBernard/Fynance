# Plan — §4 Audit des dépendances

> Plan jetable à la racine (à archiver). Tâche roadmap §4.

## Audit (usage réel dans fynance/, hors tests)
| dep | fichiers | verdict |
|-----|---------:|---------|
| numpy | 33 | cœur — garder |
| torch | 14 | backend ML — garder |
| matplotlib | 9 | plotting backtest — garder |
| polars | 5 | entrées (remplace pandas) — garder |
| scipy | 4 | optim/clustering (alloc) — garder |
| seaborn | 3 | palettes plotting — garder |
| numba | 1 | `@njit` (filters) — garder |
| **xgboost** | **0** | **inutilisé → retiré** |

## Action
- `xgboost>=2.0` retiré de `pyproject.toml` + `requirements.txt` (jamais importé ;
  dépendance lourde). Non-breaking (fynance ne l'importe nulle part). Mention
  obsolète retirée de `01-overview.md`.
- Versions plancher (`>=`) des autres deps cohérentes ; rien d'autre à changer.

## Vérif
- `import fynance` OK ; suite + ruff + mypy verts.
