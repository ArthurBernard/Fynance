# Plan — §6.2 Migrer / fermer les TODO inline

> Plan jetable à la racine (à archiver). Tâche roadmap §6.

## Constat
16 marqueurs `# TODO`/`# FIXME` hors tests. Aucun n'était assez pertinent pour
migrer vers la roadmap (déjà fournie) ; ils étaient vagues, vides, ou décidés
contre par un ADR.

## Actions
- **Retirés** (bruit/vague/décidé) : `_wrappers` (check if working), `stats`/`ratios`
  (efficiency), `allocation` (`# TODO cython` — contraire à l'ADR « no new Cython » ;
  + verify efficiency docstring), `loss` (ndarray inheritance musing), `scale`
  (axis-wrapper block), `momentums` (order-3/4 idea), `dynamic_plot_backtest`
  (FINISH DOCSTRING — docstring en fait complète ; FIXME params), `_base`
  (verify dtype), `drawdown` (cython-or-not — décidé contre).
- **Mis à jour** : `plot.py` `pandas`→`polars` (pandas retiré).
- **`momentums_cy.pyx:26` FIXME « window = w+1 »** : **vérifié périmé** — les
  property tests (#65) prouvent que `sma`/`wma`/… utilisent bien une fenêtre `w`.
  FIXME retiré, `.pyx` recompilé (`.c` régénéré committé).
- **Gardé** : `metrics_cy.pyx:967` (« why i did this? ») — incertitude réelle en
  zone Cython gelée, non résoluble sans risque ; laissé en l'état.

## Vérif
- property tests verts, suite complète 294, ruff + mypy 0, rebuild ext OK.
