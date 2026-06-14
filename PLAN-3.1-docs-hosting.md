# Plan — §3.1 Décision hébergement doc

> Plan jetable à la racine (à archiver). Tâche roadmap §3.1.

## Évaluation
État réel constaté : `.readthedocs.yaml` complet (build ext + `.[doc]` + sphinx),
badge + liens README → `fynance.readthedocs.io`, **aucun** workflow GitHub Pages.
→ **RTD est déjà l'hôte canonique** ; la note « GitHub Pages today » était périmée.

## Décision (= roadmap §3.1)
- **RTD retenu** (rien à migrer). ADR ajouté dans `03-decisions.md`.
- `.readthedocs.yaml` : `fail_on_warning: false` → **true** (cohérent avec le gate
  CI `sphinx-build -W`, déjà vert). Risque faible (mêmes deps `.[doc]`).
- Note d'hébergement périmée retirée de `06-status.md`.

## Vérif
- `sphinx-build -W` local vert (RTD `fail_on_warning: true` passera).
