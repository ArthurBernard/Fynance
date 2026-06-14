# Plan — §5.8 Backtest réaliste

> Plan jetable à la racine (à archiver). Roadmap §5.8 (bloc 🟢).

## Livré
- **Métriques de robustesse** (`features/stats.py`, exposées via le shim metrics) :
  `percent_positive` (hit rate), `tail_ratio` (|q95|/|q05|).
- **`fynance/algorithms/sizing.py`** (nouveau) :
  - `kelly_fraction(returns, fraction)` — levier Kelly fractionnel (μ/σ²).
  - `vol_target(X, target_vol, period, w, max_leverage)` — levier ciblant une vol
    constante, **causal** (réutilise `realized_volatility`), capé.
  - `transaction_cost(weights, fee)` — coût par pas basé sur le turnover |Δw|.

## Tests (10)
robustesse (formule/quantiles/2D) ; kelly (formule + var nulle) ; vol_target
(forme + cap + **non-lookahead**) ; transaction_cost (turnover 1D/2D).

## Vérif
- 10 tests + doctests (kelly, transaction_cost) ; suite 346 ; ruff + mypy 0.
