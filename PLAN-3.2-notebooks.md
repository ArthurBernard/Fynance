# Plan — §3.2 Notebooks Keras → PyTorch

> Plan jetable à la racine (à archiver). Tâche roadmap §3.2.

## Constat
3 notebooks volumineux (785 Ko–1,2 Mo, pleins de sorties sauvegardées) :
- `Exemple_Rolling_NeuralNetwork.ipynb` — Keras (7 hits).
- `test_roll_NN_test.ipynb`, `Test_various_NN_models_with_simulated_data.ipynb`
  — vieux notebooks de dev (pas de Keras mais obsolètes).

## Action
- **Remplacés** par **un** notebook propre `Notebooks/pytorch_examples.ipynb**
  (sans sorties → diff léger) couvrant : métriques, allocation ERC, modèles
  MLP/TCN/Transformer entraînés avec `SharpeLoss`, walk-forward `cross_validate`,
  losses custom. **Exécuté via le code des cellules → zéro erreur, zéro warning**
  (`ipykernel` absent de l'env, vérif par exec direct des cellules).
- **Bugs README trouvés en route et corrigés** :
  - `fy.ERC(cov)()` → `fy.ERC(cov)` (ERC renvoie un ndarray, pas un callable).
  - exemple rolling `model(n=, s=, r=)` → `model(train_period=, test_period=,
    roll_period=)` (vraie signature de `__call__`) + itération `for eval_set,
    test_set in model`.

## Vérif
- code du notebook exécuté sans erreur ni warning ; suite 317 ; ruff OK.
