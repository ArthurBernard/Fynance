# Plan — §5.3 Régimes de marché

> Plan jetable à la racine (à archiver). Roadmap §5.3 (bloc 🟢).

## Livré (`fynance/features/regime.py`)
- `detect_regimes(X, n_regimes, w, period, seed)` — labels de régime par k-means
  (scipy) sur 2 features (vol réalisée glissante + return moyen glissant),
  standardisées. Labels **ordonnés par vol croissante** (0=calme … n-1=volatil).

## Note causalité
k-means **in-sample** (voit toute la série) → outil d'**analyse** / étude de
conditionnement, pas une feature strictement causale. L'assignation online
causale (fit sur le passé) est l'extension 🟡.

## Tests (3)
forme + range des labels ; ordonnancement par vol (calme < volatil) ; déterminisme (seed).

## 🟡 Non fait (besoin de données / expériences)
- Conditionner l'architecture (mixture-of-experts / embedding de régime).
- Évaluer si la détection améliore le Sharpe out-of-sample.

## Vérif
- 3 tests ; suite 380 ; ruff + mypy 0.
