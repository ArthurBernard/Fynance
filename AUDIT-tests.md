# Audit — Tests & couverture

[← retour à l'index](AUDIT.md)

## Résumé

- **249 tests** collectés, **tous passants** + doctests (`--doctest-modules`), en
  5,07 s. 204 fonctions `test_*` réparties en miroir du package.
- **Couverture globale 78 %** (`pytest --cov`, 3770 stmts, 817 manquants).
- 5 warnings runtime (non bloquants) : fenêtres laggées hors-bornes ×4
  (`_wrappers.py`), tenseur NumPy non-writable ×1 (`_base.py:341`).
- Benchmarks `pytest-benchmark` actifs (kalman ~1,52 ms, RTS ~1,47 ms).

La discipline est réelle : tests en miroir, fixtures + `assert`, doctests traités
comme des tests, `conftest.py` qui fige le repr NumPy-2 (`legacy='1.25'`).

## 1. 🔴 Fichier de test jamais collecté <a id="1"></a>

`fynance/tests/core/series.py` **est** un fichier de test :
```python
""" Test of series objects. """
from unittest.mock import MagicMock
import pytest
from fynance.core.series import Series
```
…mais il **n'est pas nommé `test_*.py`**, donc pytest ne le collecte jamais
(`testpaths=["fynance"]` + convention de découverte par défaut). Conséquence :
`fynance/core/series.py` est à **0 % de couverture** alors qu'une suite de tests
existe et dort. Renommer en `test_series.py` débloque immédiatement ces tests
(et révélera peut-être des régressions accumulées depuis 2021).

> Note : la doc dev affiche `tests/core/series.py` à « 0 % » dans le tableau de
> couverture sans relever que c'est un test non collecté — la dérive est passée
> inaperçue.

## 2. Décalage doc vs réel

`doc/dev/01,05,06` annoncent « **~198 tests** ». Le réel est **249**. La doc a
sous-estimé / vieilli. À resynchroniser (`/groom-docs`).

## 3. Trous de couverture significatifs <a id="trous"></a>

Modules à couverture faible ou nulle (hors plotting, intrinsèquement dur à tester) :

| Module | Couv. | Commentaire |
|--------|------:|-------------|
| `core/series.py` | **0 %** | test existant non collecté (finding #1) |
| `algorithms/browsers.py` | **0 %** | aucun test ; rôle à clarifier |
| `models/basis.py` | **0 %** | code mort (à supprimer, pas à tester) |
| `backtest/_basis_plot.py` | **0 %** | plotting |
| `core/__init__.py` | **0 %** | non importé par `fynance/__init__.py` |
| `algorithms/rolling.py` | 22 % | walk-forward alloc peu testé |
| `backtest/plot_backtest.py` | 22 % | plotting |
| `backtest/plot.py` | 24 % | plotting |
| `estimator/estimator.py` | 26 % | le module « not yet working » |
| `models/econometric_models.py` | 54 % | ARMA/GARCH partiellement testé |
| `models/rolling.py` | 64 % | base walk-forward (cœur causal !) |

**Priorités de test** (par valeur/risque, pas par % brut) :
1. `models/rolling.py` (64 %) — c'est l'invariant causal cœur ; les branches non
   couvertes (`319, 333-346, 359-363, 384-432…`) incluent la boucle d'entraînement.
2. `algorithms/rolling.py` (22 %) — walk-forward allocation, même criticité causale.
3. `core/series.py` — gratuit (renommer le fichier #1).
4. `models/econometric_models.py` (54 %) — chemins ARMA/GARCH.

Le plotting (`backtest/plot*`) à faible couverture est moins prioritaire : difficile
à tester, faible risque de correction silencieuse.

## 4. Manque : test différentiel py ↔ cy

La double implémentation Cython/Python (cf. AUDIT-architecture §2.1) n'a pas, de
façon systématique, de test qui **compare** la sortie `metrics.py` vs
`metrics_cy.pyx` sur les mêmes entrées. C'est exactement le filet qui aurait
attrapé le FIXME `momentums_cy.pyx:26` (fenêtre `w+1`). Recommandation : une suite
paramétrée `assert np.allclose(py_impl(x), cy_impl(x))` par paire.

## 5. Conformité à l'invariant causal

`doc/dev/05` exige un « audit causal » pour toute feature rolling (pas de
`center=True`, pas de normalisation globale). C'est une convention forte et juste.
Je n'ai pas trouvé de **test générique** qui vérifie automatiquement l'absence de
lookahead (p.ex. : perturber `X[t+1:]` et vérifier que `f(X)[t]` est inchangé). Ce
serait un test de propriété de grande valeur pour une lib de backtest — aujourd'hui
l'invariant repose sur la revue humaine.

## 6. Verdict tests

**B+**. Suite saine, rapide, doctests inclus, bonne discipline. Trois axes :
(1) corriger le fichier non collecté #1, (2) renforcer la couverture du cœur causal
(`*/rolling.py`), (3) ajouter des tests de propriété (parité py/cy + non-lookahead)
qui transformeraient des conventions humaines en garde-fous exécutables.
