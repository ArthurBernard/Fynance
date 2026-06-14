# Audit — Code : fonctions, objets, qualité

[← retour à l'index](AUDIT.md)

Périmètre : 562 fonctions, 61 classes, 15 651 LOC (hors tests). `ruff check` passe
sans erreur — la base de style est propre. Les problèmes ci-dessous sont
sémantiques, pas stylistiques.

## 1. Code mort <a id="mort"></a>

### `models/basis.py` — stubs jamais finis
- `SignalModel.get_signal()` retourne `self._get_signal(self.y_pred)` mais
  **`self.y_pred` n'est jamais assigné** nulle part (`basis.py:55`) → toute
  utilisation lève `AttributeError`.
- `MagnitudeModel` est une classe vide (`pass`, `basis.py:58-61`).
- `__all__ = []` → non exporté, 0 % de couverture.
- La doc dev le note déjà comme « slated for removal » (`doc/dev/04`). **À
  supprimer.**

### `backtest/dynamic_plot_backtest.py` — versions périmées
- `__BacktestNeuralNet` marqué `# OLD VERSION => DEPRECIATED` (`:143`).
- `_BacktestNeuralNet` stub inutilisé, `# TODO : to implement base class` (`:600`).
- Fichier de 708 lignes / 6 classes à 29 % de couverture. **À élaguer.**

## 2. Gestion d'erreur défaillante <a id="estimator"></a>

`estimator/estimator.py` :
```python
else:
    print('Unknow model.')   # faute : "Unknow" ; print au lieu de message
    raise ValueError          # exception SANS message
```
Apparaît **deux fois** (`:52` et `:76`). Correction triviale :
`raise ValueError(f"Unknown model: {model!r}")`.

Pire, la fonction `estimation()` elle-même est annoncée cassée :
```python
def estimation(y, x0, p=0, q=0, Q=0, P=0, cons=True, model='arch'):
    """
    NOT YET WORKING !
    Estimator
    """
    ...  # NEED TO FIND AN OPTIMIZER
```
(`estimator.py:35-57`). Un symbole public à moitié implémenté laissé dans une lib
`Production/Stable` est un piège pour l'utilisateur. **Décision à prendre** : finir,
ou retirer/marquer expérimental explicitement.

## 3. `print()` de debug en code de lib <a id="prints"></a>

`models/rolling.py:354-363` — dans un `except` qui re-`raise` :
```python
except Exception as e:
    print(train_slice)
    print(self.X[train_slice])
    print(self.f(self.y[train_slice]))
    raise e
```
Dump d'arrays brutes sur stdout en cas d'erreur d'entraînement. À remplacer par du
`logging.debug` ou à supprimer (le `raise e` perd au passage la stacktrace
d'origine — préférer `raise`).

Autres `print` en flux de contrôle (légitimes mais à terme à passer en `logging`) :
- `algorithms/allocation.py:462` `print(mat_cov)` (debug ?), `:765` message d'itérations dépassées.
- `algorithms/rolling.py:78`, `models/rolling.py:458` : barres de progression `end='\r'` — acceptable pour une lib interactive, mais non configurable.

## 4. Annotations de type mensongères <a id="typage"></a>

`mypy fynance/` → **104 erreurs / 15 fichiers**. La plupart sont du bruit
NumPy-2 (`Returning Any`), mais une catégorie est un vrai défaut d'API :

`algorithms/allocation.py` déclare des signatures `NDArray[np.float64]` alors que
`rolling_allocation()` manipule en réalité des **DataFrames** :
```
allocation.py:687: "ndarray[...]" has no attribute "index"
allocation.py:688: ... has no attribute "columns"
allocation.py:692: ... has no attribute "pct_change"
allocation.py:719: ... has no attribute "loc"
```
Les annotations *contredisent* l'usage réel : un utilisateur qui se fie aux types
sera induit en erreur, et l'IDE ne l'aidera pas. À corriger en
`Union[NDArray, pd.DataFrame]` ou via `@overload`. C'est le finding de typage le
plus actionnable.

`backtest/print_stats.py:90-136` : ~10 erreurs « bool is not indexable » →
réutilisation d'une variable typée `bool` puis réassignée en `ndarray` : code qui
marche mais sémantiquement confus.

## 5. Bug connu non corrigé (FIXME) <a id="fixme"></a>

`features/momentums_cy.pyx:26` : `# FIXME : problem with window size => window is
w + 1 instead of w`. C'est un **bug de correction silencieux** : les moyennes
mobiles Cython utiliseraient une fenêtre de `w+1` au lieu de `w`. Les warnings
runtime observés en test (`lagged window of size 7 out of bounds with time axis of
size 6`, `_wrappers.py:249`) pourraient être reliés. À traiter avec un test de
non-régression comparant explicitement la fenêtre py vs cy.

## 6. Inventaire des TODO/FIXME <a id="todos"></a>

23 marqueurs hors tests. Les plus parlants :

| Fichier:ligne | Marqueur |
|---|---|
| `estimator/estimator.py` | "NOT YET WORKING", "NEED TO FIND AN OPTIMIZER" |
| `features/momentums_cy.pyx:26` | fenêtre `w+1` (bug) |
| `features/metrics_cy.pyx:967` | "why i did this ? To fix" |
| `backtest/dynamic_plot_backtest.py:38` | "FINISH DOCSTRING" |
| `backtest/dynamic_plot_backtest.py:121` | params "not explicitly defined and not saved" |
| `algorithms/allocation.py:151` | "verify the efficiency" |
| `_wrappers.py:181` | "check if it's working" |
| `features/metrics.py` ×5 | "check efficiency", "make cython function or not", "rolling perf metric" |

Aucun n'est bloquant, mais leur densité dans `metrics`, `dynamic_plot_backtest` et
`estimator` confirme ces trois zones comme les plus « brutes ».

## 7. Dépréciation propre (point positif)

`features/indicators.py:144-152` : `bollinger_band` émet un `DeprecationWarning`
correct (message clair, `stacklevel=2`, échéance « fynance 2.0 ») pour l'ancien
chemin de retour mono-array. C'est le **bon modèle** de gestion de dépréciation,
cohérent avec la politique 1.x annoncée dans `fynance/__init__.py`.

## 8. Métadonnées de version désynchronisées <a id="version"></a>

`fynance.__version__` retourne **`1.2.0`** (via `importlib.metadata.version`,
`__init__.py:56-61`) car `fynance.egg-info/PKG-INFO` est resté à 1.2.0, alors que
`pyproject.toml` est à **1.3.4**. Symptôme d'un editable install non reconstruit.
Sans gravité pour les utilisateurs PyPI (le wheel publié porte la bonne version)
mais trompeur en dev. `pip install -e .` pour réaligner.

## 9. Qualité objets (positif)

- `models/loss/` : `BaseLoss(nn.Module)` + sous-classes `SharpeLoss/SortinoLoss/
  DirectionalAccuracyLoss`, entièrement typées (`float`, `int`, `object`), 100 % de
  couverture, `_check_tensor` de validation. **Référence de qualité du repo.**
- `core/series.py` `Series(np.ndarray)` : sous-classe propre (`__new__` +
  `__array_finalize__` corrects). Seul souci : non testée (voir AUDIT-tests).
- Pas de mutable default args, pas de `except:` nu, pas d'`eval/exec`, pas
  d'`assert` en code de prod. Hygiène de base correcte.
