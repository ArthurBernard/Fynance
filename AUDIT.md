# Fynance — Audit complet du dépôt

> Audit réalisé le **2026-06-14** sur la branche `develop` (HEAD `f95e88d`).
> Méthode : exploration structurelle → métriques objectives (pytest, ruff, mypy,
> interrogate, coverage) → lecture du code source → recoupement avec la doc.
> Toutes les affirmations sont étayées par des références `fichier:ligne` ou des
> sorties d'outils reproductibles.

Ce fichier est l'**index + synthèse globale**. Les analyses détaillées par angle :

| Angle | Fichier |
|-------|---------|
| Architecture & design | [`AUDIT-architecture.md`](AUDIT-architecture.md) |
| Code : fonctions, objets, qualité | [`AUDIT-code.md`](AUDIT-code.md) |
| Tests & couverture | [`AUDIT-tests.md`](AUDIT-tests.md) |
| Documentation | [`AUDIT-docs.md`](AUDIT-docs.md) |

---

## 1. Synthèse exécutive

**fynance** est une bibliothèque numérique Python + Cython mature (premier commit
2018, 620 commits, v1.3.4 sur PyPI, `Production/Stable`) pour l'analyse de séries
financières : features, allocation de portefeuille, modèles économétriques/neuronaux,
backtest. L'architecture en couches est saine, l'invariant de causalité (no
lookahead) est clair et défendu, et le socle numérique (`features`, `algorithms`,
`estimator`) est correct et testé.

Le projet est **globalement en bon état**, mais porte une **dette technique
héritée nettement identifiable** : code mort (stubs jamais retirés), un module
économétrique Python annoncé « not yet working », des annotations de type qui
mentent (104 erreurs mypy), une CI qui n'applique ni mypy ni interrogate, et une
documentation `doc/dev/` qui a légèrement dérivé du code réel. Aucun de ces points
n'est bloquant ; ce sont des nettoyages à coût modéré et fort retour sur lisibilité.

### Note de santé par dimension

| Dimension | Note | Justification courte |
|-----------|:----:|----------------------|
| Architecture | **A-** | Couches nettes, invariant causal explicite ; quelques gros modules à scinder |
| Qualité du code | **B** | ruff clean, mais code mort, prints de debug, `print`+`raise` sans message |
| Typage | **C** | mypy configuré mais 104 erreurs, annotations fausses dans `allocation.py` |
| Tests | **B+** | 249 tests passent + doctests ; 1 fichier de test non collecté, trous de couverture (78%) |
| Documentation | **A-** | Docstrings numpydoc 88.6%, doc dev riche ; quelques dérives factuelles |
| CI / Process | **B+** | Matrice 3.10-3.13, wheels, release ; mais mypy/interrogate hors-gate |
| **Global** | **B+** | Socle solide et publié, dette héritée circonscrite et peu coûteuse à solder |

---

## 2. Chiffres clés (mesurés, pas estimés)

| Métrique | Valeur | Source |
|----------|-------:|--------|
| Fichiers `.py`/`.pyx` (hors tests) | 69 | `find` |
| LOC totales (py+pyx) | 15 651 | `wc -l` |
| Fonctions (def/cpdef, hors tests) | 562 | `grep` |
| Classes (hors tests) | 61 | `grep` |
| Fonctions de test | 204 | `grep def test_` |
| Tests collectés & passants | **249** (+ doctests) | `pytest`, 5.07 s |
| Couverture globale | **78 %** | `pytest --cov` |
| Couverture docstrings (interrogate) | **88.6 %** | `interrogate` |
| Erreurs ruff | **0** | `ruff check` |
| Erreurs mypy | **104** (15 fichiers) | `mypy fynance/` |
| Warnings runtime en test | 5 | `pytest` |
| Fichiers à en-tête 2018-2021 (legacy) | 26 | `grep @Date` |

---

## 3. Findings priorisés (le tableau actionnable)

Sévérité : 🔴 à corriger · 🟠 important · 🟡 nettoyage · 🔵 amélioration.

| # | Sév. | Finding | Emplacement | Détail |
|---|:----:|---------|-------------|--------|
| 1 | 🔴 | **Fichier de test jamais collecté** : `series.py` au lieu de `test_series.py` → `core/series.py` n'est testé par *rien* (0 % couv.) alors qu'un test existe | `fynance/tests/core/series.py` | [tests](AUDIT-tests.md) |
| 2 | 🔴 | **Annotations de type mensongères** : `rolling_allocation` & co déclarent `NDArray[float64]` mais manipulent des DataFrames (`.index/.columns/.pct_change`) | `algorithms/allocation.py:626-727` | [code](AUDIT-code.md) |
| 3 | 🟠 | **Module économétrique Python « NOT YET WORKING »** exposé : `estimation()` a un docstring d'aveu + commentaire « NEED TO FIND AN OPTIMIZER » | `estimator/estimator.py:35-57` | [code](AUDIT-code.md) |
| 4 | 🟠 | **mypy hors CI** : 104 erreurs jamais bloquées ; le type-checking ne protège rien | `.github/workflows/ci.yml` | [archi](AUDIT-architecture.md) |
| 5 | 🟠 | **`print()` de debug oublié** dans un `except` qui re-`raise` (dump d'arrays sur stdout) | `models/rolling.py:360-362` | [code](AUDIT-code.md) |
| 6 | 🟠 | **`print('Unknow model.')` + `raise ValueError`** sans message (×2, faute de frappe) au lieu d'une exception parlante | `estimator/estimator.py:52,76` | [code](AUDIT-code.md) |
| 7 | 🟡 | **Code mort jamais retiré** : `SignalModel` utilise `self.y_pred` jamais défini ; `MagnitudeModel` = `pass` | `models/basis.py` | [code](AUDIT-code.md) |
| 8 | 🟡 | **Code mort backtest** : `__BacktestNeuralNet` « OLD VERSION => DEPRECIATED » + `_BacktestNeuralNet` stub | `backtest/dynamic_plot_backtest.py:143,600` | [code](AUDIT-code.md) |
| 9 | 🟡 | **Bug connu non corrigé (FIXME)** : taille de fenêtre `w+1` au lieu de `w` dans les momentums Cython | `features/momentums_cy.pyx:26` | [code](AUDIT-code.md) |
| 10 | 🟡 | **Doc dev dérive du réel** : annonce « ~198 tests » (réel 249), oublie `models/loss/`, `_exceptions.py`, `money_management.py` | `doc/dev/01,05,06` | [docs](AUDIT-docs.md) |
| 11 | 🟡 | **interrogate hors CI** : seuil 80 % seulement pour le badge, pas gate | `.github/workflows/badges.yml` | [docs](AUDIT-docs.md) |
| 12 | 🟡 | **Métadonnées installées périmées** : `fy.__version__` == `1.2.0` (egg-info) vs `1.3.4` (pyproject) | `fynance.egg-info` | [code](AUDIT-code.md) |
| 13 | 🔵 | **Gros modules à scinder** : `features/metrics.py` (1782 l.), `metrics_cy.pyx` (1210), `allocation.py` (767), `dynamic_plot_backtest.py` (708, 6 classes) | — | [archi](AUDIT-architecture.md) |
| 14 | 🔵 | **Trous de couverture** : `core/series.py` 0 %, `algorithms/browsers.py` 0 %, `algorithms/rolling.py` 22 %, `estimator/estimator.py` 26 %, backtest/plot* 14-29 % | — | [tests](AUDIT-tests.md) |
| 15 | 🔵 | **Notebook Keras résiduel** vs politique « PyTorch only » | `Notebooks/Exemple_Rolling_NeuralNetwork.ipynb` | [docs](AUDIT-docs.md) |
| 16 | 🔵 | **23 marqueurs TODO/FIXME** dispersés (efficiency, docstrings inachevés, « why i did this ? ») | tout le code | [code](AUDIT-code.md) |

---

## 4. Recommandations — ordre suggéré

**Quick wins (quelques minutes chacun, fort signal)**
1. Renommer `tests/core/series.py` → `test_series.py` (#1) — débloque tout un pan de tests existant mais dormant.
2. Remplacer `print('Unknow model.'); raise ValueError` par `raise ValueError(f"Unknown model: {model!r}")` (#6).
3. Retirer les 3 `print()` de debug dans `models/rolling.py` (#5).
4. Rebuild de l'install editable pour aligner `__version__` (#12).

**Nettoyage de dette (1 PR ciblée chacun, conforme au Git Flow du repo)**
5. Supprimer le code mort `models/basis.py` + stubs backtest (#7, #8) — déjà tracké comme « slated for removal » dans la doc dev.
6. Corriger les annotations de type de `allocation.py` (Union ndarray|DataFrame ou surcharge), puis **ajouter mypy + interrogate à la CI** (#2, #4, #11).
7. Traiter ou retirer `estimator/estimator.py` (#3) : soit le finir, soit le marquer explicitement non-public/expérimental.

**Fond (épics, via le dev-loop `/pick-task → /plan`)**
8. Scinder les gros modules (#13) et combler les trous de couverture (#14).
9. Réécrire les notebooks en PyTorch (#15), corriger le FIXME momentums avec test de non-régression (#9).
10. Synchroniser `doc/dev/` (`/groom-docs`) — chiffres, modules oubliés (#10).

---

## 5. Points forts à préserver

- **Invariant de causalité** explicite, documenté (`doc/dev/03`) et testé — c'est la valeur cœur d'une lib de backtest, et elle est prise au sérieux.
- **API publique gelée 1.x** clairement énoncée dans `fynance/__init__.py:36-52`, avec chemin de dépréciation (`DeprecationWarning` promu en erreur pour le namespace interne via `pyproject.toml:79-81`).
- **Pas de collision de noms** dans le namespace plat (115 symboles exportés, vérifié programmatiquement).
- **Module `models/loss/`** : code récent, propre, entièrement typé et testé à 100 % — le modèle de ce que devrait être le reste.
- **Doctests = tests** : discipline réelle (`--doctest-modules`), `conftest.py` gère le repr NumPy-2.
- **Build robuste** : fallback Cython `.pyx → .c`, wheels manylinux, matrice 3.10-3.13.
