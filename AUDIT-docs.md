# Audit — Documentation

[← retour à l'index](AUDIT.md)

La documentation de fynance est **plurielle et globalement de bonne qualité** :
docstrings numpydoc, doctests exécutables, site Sphinx (furo), un pack
d'orientation `doc/dev/` pour agents, README, CHANGELOG, CONTRIBUTING. Les défauts
sont des **dérives factuelles** et quelques angles morts, pas un manque de doc.

## 1. Docstrings (code)

- **Couverture interrogate : 88.6 %** (220 objets, 25 sans docstring ; seuil
  configuré 80 %). Bon niveau.
- Style numpydoc cohérent (Parameters / Returns / References / Examples), avec
  formules LaTeX (`r"""`) dans `models/basis.py`, `models/loss/`, etc.
- Trous résiduels (interrogate `-v`) : `backtest/plot_tools.py` (50 %),
  `backtest/print_stats.py` (50 %), `models/gru.py` (67 %), `models/lstm.py` /
  `models/rolling.py` (75 %).
- Docstrings **inachevés** signalés en clair : `dynamic_plot_backtest.py:38`
  `# TODO : FINISH DOCSTRING`.
- **Doctests = tests** : exécutés via `--doctest-modules`, donc garantis corrects
  (un exemple cassé fait échouer la CI). C'est une vraie force — la doc d'API ne
  peut pas pourrir en silence.

## 2. Site Sphinx (`doc/source/`)

- Stack moderne : furo, `sphinx-design`, `sphinx-copybutton`, `nbsphinx`,
  `numpydoc`, `viewcode`, autosummary (`generated/` peuplé : ERC/HRP/IVP/MDP/MVP/
  MVP_uc, backtest, …).
- Pages dédiées : `installation`, `quickstart`, `changelog`, + une `.rst` par
  sous-module. Harmonisation récente (header collant, logos light/dark, hero) —
  cf. CHANGELOG 1.3.4.
- **Manque CI** : pas de job `make html` dans `ci.yml` (cf. AUDIT-architecture
  §4). Une rupture Sphinx ne serait détectée qu'au build RTD.

## 3. Doc d'orientation agent (`doc/dev/`)

Pack `01-overview` → `07-roadmap` + `03-decisions` (journal ADR) + plans. **Très
bonne initiative** : rationale, matrice de stabilité par sous-package, méthodologie
de test, statut. Recoupé au code, c'est largement fidèle. Mais quelques dérives :

### 3.1 Dérives factuelles <a id="derive"></a>

| Affirmation doc | Réalité | Où |
|---|---|---|
| « ~198 tests » | **249** | `01-overview`, `05-testing`, `06-status` |
| Liste des sous-packages features | oublie **`money_management.py`** (présent, 36 % couv.) | `01`, `02` |
| Utilitaires | ne mentionne pas **`_exceptions.py`** (existe, `ArraySizeError`, 100 % couv.) | `01-overview` |
| `models/` modules | ne liste pas explicitement le sous-package **`models/loss/`** dans le repo map (mentionné en prose ailleurs) | `01-overview` repo map |
| Tableau couverture cite `tests/core/series.py` à 0 % | …sans signaler que c'est un **test non collecté** (bug, pas absence de test) | implicite |

Aucune n'est grave, mais elles montrent que `doc/dev/` n'a pas été re-synchronisé
après l'ajout des tests/losses. Un passage `/groom-docs` le corrigerait.

### 3.2 Cohérence du discours « PyTorch only »
La doc affirme « Keras/TensorFlow retiré, pas étendu ». **Le code source est bien
0 % Keras/TF** (vérifié : aucun `import keras/tensorflow` dans `fynance/`). En
revanche un **notebook conserve du Keras** : `Notebooks/Exemple_Rolling_
NeuralNetwork.ipynb`. La doc le reconnaît (`06-status` : « Notebooks still carry
Keras examples — to be rewritten »). Cohérent mais à solder.

## 4. README & docs de premier contact

- **README.md** : clair, badges, install, quickstart (Sharpe, ERC, rolling NN),
  liens. Le snippet `RollMultiLayerPerceptron(X, y, layers=[64,32])` suppose `X,y`
  définis — exemple illustratif, non exécuté (donc non garanti par doctests).
- **CHANGELOG.md** : format Keep a Changelog rigoureux, SemVer, sections
  `[Unreleased]` tenues, références PR (#52, #53, #57). Très bien tenu.
- **CONTRIBUTING.md**, **CLAUDE.md** présents (CLAUDE.md gitignored — local).

## 5. Angles morts documentaires

- Pas de doc sur le module `estimator/estimator.py` « not yet working » — un
  utilisateur ne sait pas qu'il ne doit pas l'utiliser (cf. AUDIT-code §2).
- `algorithms/browsers.py` (0 % couv., non documenté) : rôle non explicité ni dans
  la doc dev ni dans le site.
- Pas de guide « architecture decision » exposé côté utilisateur (réservé à
  `doc/dev/`, gitignoré pour la roadmap) — choix de privacy assumé.

## 6. Verdict documentation

**A-**. Docstrings solides et *testées*, site Sphinx soigné, doc agent riche. À
corriger : resynchroniser les chiffres et inventaires de `doc/dev/` (#10),
documenter/retirer les zones « brutes » (`estimator.py`, `browsers.py`), ajouter un
gate `make html` en CI, et réécrire le notebook Keras restant.
