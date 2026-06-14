# Plan — §5.4 Features / indicateurs techniques

> Plan jetable à la racine (à archiver). Roadmap §5.4 (bloc 🟢).

## Déjà présents (non re-faits)
Bollinger, MACD (line/hist/signal), RSI, CCI, HMA ; EMA/SMA/WMA + std.

## Ajoutés (single-série, causaux, `@WrapperArray`, dans `indicators.py`)
- `roc` — Rate of Change (momentum).
- `realized_volatility` — std glissante des log-returns, annualisée.
- `rolling_skewness`, `rolling_kurtosis` (via scipy.stats, fenêtre stricte passée).
- `rolling_autocorr` — autocorrélation glissante lag-k.

15 tests : parité vs référence/scipy + **non-lookahead** + colonne-par-colonne 2D.

## Différé (nécessite une API multi-séries OHLCV)
ATR, ADX, Williams %R (High/Low), OBV, VWAP (Volume) — demandent High/Low/Volume,
incompatibles avec la convention single-série actuelle. Décision d'API à part.
GARCH-as-feature : possible via `fynance.estimator` mais coûteux ; reporté.

## Vérif
- 15 tests verts ; suite 333 ; ruff + mypy 0.
