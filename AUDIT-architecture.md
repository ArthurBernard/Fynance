# Audit — Architecture & design

[← retour à l'index](AUDIT.md)

## 1. Vue d'ensemble

fynance est une **bibliothèque numérique en couches**, pas un service : pas d'I/O,
pas de daemon. Les appelants passent des arrays/DataFrames et récupèrent des
arrays/modèles. Le découpage est par *concern*, ce qui est sain et lisible :

```
backtest            évaluation + tracé des résultats
   │ consomme
   ▼
algorithms · models  allocation, walk-forward training, économétrie
   │ s'appuie sur
   ▼
features · estimator indicateurs, métriques, params ARMA/GARCH (hot paths Cython)
   │ opère sur
   ▼
core                 helpers de séries, wrappers d'arrays
```

Cette structure est correcte et bien documentée dans `doc/dev/02-architecture.md`.
Les dépendances vont dans le bon sens (pas de cycle évident entre couches au niveau
package). Le namespace public plat (`import fynance as fy`) expose **115 symboles**
sans collision — vérifié programmatiquement.

## 2. Les trois patterns transverses (et leur état)

### 2.1 Double implémentation Cython / Python (`features/`)
Chaque hot path existe deux fois : `metrics_cy.pyx` (compilé `.so`) + `metrics.py`.
Le `.py` fait `from fynance.features.metrics_cy import *` (`metrics.py:40`) puis
ajoute des fonctions pures-Python par-dessus. Le garde `USE_CYTHON='auto'` de
`setup.py` compile le `.pyx` si Cython présent, sinon retombe sur le `.c` livré.
**Pattern sain, fallback robuste.** La règle « nouveau code numérique = Numba
`@njit`, pas de nouveau Cython » est claire (`doc/dev/03`).

> Réserve : ce double-fichier double aussi la surface de bug (cf. le FIXME
> `momentums_cy.pyx:26` sur la taille de fenêtre `w+1` — voir `AUDIT-code.md`). La
> parité py/cy n'est pas testée différentiellement de façon systématique.

### 2.2 Rolling / walk-forward (le cœur causal)
`_RollingBasis` (`models/rolling.py`) est un **itérateur** : `__call__` fixe la
fenêtre (`n` train, `s` test, `r` roll), chaque `__next__` entraîne sur `X[t-n:t]`
et prédit sur `X[t:t+s]`. `RollMultiLayerPerceptron` en hérite ;
`rolling_allocation()` (`algorithms/`) reprend la même forme en décorateur. C'est
**le** garde-fou anti-lookahead, et il est posé comme invariant structurel
(`doc/dev/03` : « a single leaked future value invalidates a backtest »). Excellent
choix de design.

### 2.3 Estimator → models
`estimator/estimator_cy.pyx` estime les params ARMA/GARCH ;
`models/econometric_models.py` les enveloppe via `get_parameters()`. La couche
Python ne ré-implémente pas l'estimation. Bonne séparation de responsabilité.

> Bémol majeur : `estimator/estimator.py` (le pendant Python non-Cython) contient
> une fonction `estimation()` au docstring « **NOT YET WORKING !** » +
> « NEED TO FIND AN OPTIMIZER » (`estimator.py:35-57`). Couverture 26 %. Ce module
> brouille la frontière « Cython = autoritatif » : il faut soit le finir, soit le
> marquer non-public/expérimental. Voir `AUDIT-code.md`.

## 3. Modules trop gros (dette structurelle) <a id="modules"></a>

Mesuré au `wc -l`. Au-delà de ~500 lignes, un module mélange plusieurs concerns.

| Fichier | LOC | Remarque |
|---------|----:|----------|
| `features/metrics.py` | 1782 | déjà identifié pour split (`doc/dev/04`) |
| `features/metrics_cy.pyx` | 1210 | jumeau Cython, même problème |
| `algorithms/allocation.py` | 767 | 5 méthodes d'alloc + helpers + rolling dans un fichier |
| `backtest/dynamic_plot_backtest.py` | 708 | **6 classes**, dont 2 mortes (voir code) |
| `features/indicators.py` | 656 | |
| `features/momentums_cy.pyx` | 586 | |
| `models/rolling.py` | 580 | |

Ce n'est pas urgent (la lib marche), mais ça augmente le coût de chaque
modification et le risque de régression. Le repo le sait déjà (roadmap §5). À
traiter par scissions atomiques, une par PR, en préservant l'API publique gelée.

## 4. CI / packaging <a id="ci"></a>

**Bon** :
- Matrice Python 3.10–3.13 (`ci.yml`), `fail-fast: false`.
- Build Cython explicite avant tests ; upload coverage Codecov.
- `release.yml` : wheels manylinux `cibuildwheel` + sdist + publish PyPI + GitHub
  Release sur tag `v*`.
- `pyproject.toml` est la source unique des métadonnées ; `setup.py` ne fait que le
  Cython. Version statique unique (`1.3.4`).
- `filterwarnings` promeut les `DeprecationWarning` *internes* en erreurs
  (`pyproject.toml:79-81`) — les dépréciations ne peuvent pas passer en silence.

**Manques** :
- **mypy n'est PAS dans la CI** (`ci.yml` ne lance que `ruff`). mypy est configuré
  (`pyproject.toml:104-121`) mais produit **104 erreurs** — il ne protège donc rien
  aujourd'hui. Soit on corrige + on gate, soit la config est décorative.
- **interrogate n'est PAS un gate** : il ne sert qu'à générer le badge
  (`badges.yml:35`). Le seuil `fail-under = 80` du `pyproject.toml` n'est jamais
  appliqué en PR.
- Pas de job docs (`make html`) en CI alors que c'est listé comme couche de test
  (`doc/dev/05`). Une rupture Sphinx ne serait pas attrapée.

## 5. Cohérence des dépendances

`requirements.txt` est dupliqué de `pyproject.toml` mais le déclare honnêtement
(« authoritative versions are defined in pyproject.toml », `requirements.txt:1`).
Acceptable. `requirements-dev.txt` existe en parallèle de l'extra `[dev]` — léger
risque de divergence à surveiller.

## 6. Verdict architecture

Architecture **A-**. Le squelette est juste, l'invariant cœur est défendu, le build
est robuste. Les points à corriger sont périphériques : gates CI incomplets, gros
modules hérités, et une frontière estimator Python/Cython floue. Rien ne remet en
cause le design.
