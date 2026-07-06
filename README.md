<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ArthurBernard/Fynance/develop/doc/source/_static/logo-dark-transparent.svg">
  <img alt="Fynance logo" src="https://raw.githubusercontent.com/ArthurBernard/Fynance/develop/doc/source/_static/logo-light-transparent.svg" height="180px" align="left">
</picture>

# **Fynance**

[![Python versions](https://img.shields.io/pypi/pyversions/fynance)](https://pypi.org/project/fynance/)
[![PyPI](https://img.shields.io/pypi/v/fynance.svg)](https://pypi.org/project/fynance/)
[![PyPI status](https://img.shields.io/pypi/status/fynance.svg?colorB=blue)](https://pypi.org/project/fynance/)
[![CI](https://github.com/ArthurBernard/Fynance/actions/workflows/ci.yml/badge.svg)](https://github.com/ArthurBernard/Fynance/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/ArthurBernard/fynance.svg)](https://github.com/ArthurBernard/Fynance/blob/master/LICENSE.txt)<br>
[![Documentation](https://readthedocs.org/projects/fynance/badge/?version=latest)](https://fynance.readthedocs.io/en/latest/)
[![Coverage](https://codecov.io/gh/ArthurBernard/Fynance/branch/develop/graph/badge.svg)](https://codecov.io/gh/ArthurBernard/Fynance)
[![Docstring coverage](https://raw.githubusercontent.com/ArthurBernard/Fynance/develop/badges/interrogate_badge.svg)](https://github.com/ArthurBernard/Fynance)
[![Downloads](https://pepy.tech/badge/fynance)](https://pepy.tech/project/fynance)

___

Pure-Python package (Numba-accelerated kernels) providing **machine learning**,
**econometric** and **statistical** tools for **financial analysis** and
**backtesting of trading strategies**.

## Installation

```bash
pip install fynance
```

From source:

```bash
git clone https://github.com/ArthurBernard/Fynance.git
cd Fynance
pip install -e ".[dev]"
```

The build is pure-Python — there is no compile step (numerical kernels are
Numba `@njit`, JIT-compiled on first call).

## Architecture

A complete, layered ML/DL backtesting tool — **data → features → signal →
portfolio → backtest → metrics** — composed through `typing.Protocol` seams.
numpy is the lingua franca; PyTorch is confined to `fynance.models`. Each piece
is usable standalone; `fynance.strategy.Strategy` is an *optional* orchestrator.

> **2.0 is a breaking release.** See [`doc/MIGRATION-2.0.md`](doc/MIGRATION-2.0.md)
> for the import-path map (e.g. `fynance.algorithms` → `fynance.portfolio`,
> performance metrics → `fynance.metrics`).

## Subpackages

**Core** `fynance.core` — `PriceSeries` value object (thin, numpy-backed) and the
pipeline protocols (`DataSource`, `FeatureTransform`, `SignalModel`, `Allocator`,
`CostModel`, `Metric`); executable conformance/causality checks (`check_conforms`,
`assert_causal`) and duck-typed pandas/polars seams (`from_pandas`/`to_pandas`).

**Data** `fynance.data` — file adapters (`load` for CSV/Parquet → `PriceSeries`),
alignment/resampling, no-lookahead temporal splits (`train_test_split`,
`walk_forward`, combinatorial-purged `combinatorial_purged_cv`) and vendor-agnostic
intraday session utilities (`session_mask`/`session_id`/`split_sessions`).

**Features** `fynance.features` — technical indicators (Bollinger, RSI, MACD, ROC,
realized volatility, rolling skew/kurtosis/autocorr, …), OHLCV indicators (ATR,
ADX, Williams %R, OBV, VWAP), a causal GARCH(1,1) conditional-volatility feature,
momentums (SMA, EMA, WMA) and adaptive windows, **NaN-aware cross-sectional
operators** (`cs_rank`/`cs_zscore`/`cs_neutralize`, …), **pairwise rolling stats**
(`roll_corr`/`roll_beta`, `cross_corr`), **fractional differentiation** (`fracdiff`),
**AFML labeling** (`triple_barrier`, `meta_labels`, uniqueness weights), scaling
(incl. rolling rank), statistics, feature engineering (multi-resolution, Granger
causality) and market-regime detection.

**Metrics** `fynance.metrics` — performance/evaluation metrics (Sharpe, Sortino,
Calmar, drawdown, …) and a one-call `summary`; **tail risk** (VaR/CVaR/CDaR,
tail dependence), **benchmark-relative** metrics (alpha, beta, tracking error,
information ratio, capture), **factor evaluation** (quantile returns, rolling IC,
IC decay), and turnover/exposure and per-trade analytics.

**Signal** `fynance.signal` — prediction → position mappers (`sign`, `threshold`,
`rank`, vol-targeting) and a model+mapper pipeline.

**Portfolio** `fynance.portfolio` — allocation (ERC, HRP, IVP, MDP, MVP, plus
risk-budgeting `RBP`) with an opt-in `cov=` seam over conditioned **covariance
estimators** (Ledoit-Wolf, EWMA, factor, Marchenko-Pastur denoising); risk
**attribution**, an exposure-**constraints** overlay (`project_weights`),
**rebalancing policies** (calendar/band/turnover-cap, lot discretization) and
sizing (fractional Kelly, single-series and book-level volatility targeting,
transaction costs).

**Backtest** `fynance.backtest` — vectorized engine (`backtest`: positions +
returns/prices + cost → `BacktestResult`), cost models (`ProportionalCost`, the
non-linear `MarketImpactCost`, per-bar `HoldingCost` and `CompositeCost` stacking)
and capacity analysis (`capacity_curve`, `breakeven_fee`).

**Plot** `fynance.plot` — composable matplotlib figures and one-call `tearsheet`
and `factor_tearsheet` reports.

**Models** `fynance.models` — econometric models (MA, ARMA, ARMA-GARCH, plus
GJR/EGARCH kernels) and PyTorch nets (MLP, RNN, GRU, LSTM, MultiHeadAttention,
TCN, Transformer), a direction+magnitude stacking ensemble, `RegimeMoE`
(regime-conditioned mixture-of-experts), objective-aligned training with
cross-asset pretraining, distributional `QuantileModel`, uncertainty wrappers
(`DeepEnsemble`, `MCDropout`), causal conformal intervals, purged walk-forward
hyperparameter search, differentiable losses (Sharpe, Sortino, Calmar, Omega,
directional, hybrid, pinball), and robust-training utilities.

**Estimator** `fynance.estimator` — volatility-model MLE: `fit_volatility` fits
GARCH / GJR-GARCH / EGARCH with Gaussian or Student-t innovations, returning a
`VolatilityResult` (AIC/BIC, conditional vol, forecast, simulate).

**Strategy** `fynance.strategy` — optional orchestrator composing the maillons
end-to-end, with single-run and walk-forward evaluation.

**Research** `fynance.research` — data-agnostic experiment harness: `Experiment`
(serializable run record), `run_experiment` (seeded, cost-aware, walk-forward),
`write_report` (portable markdown + tearsheet PNG + notebook), synthetic data
generators (`gbm`, `regime_switching`) and overfitting guards (permutation and
block bootstrap, deflated Sharpe, PBO/CSCV, walk-forward MDA feature importance).
Results are written only to a caller-provided `output_dir` — fynance never stores
them itself.

## Quick start

```python
import numpy as np
import fynance as fy

# 1. Data — load a CSV/Parquet file, or build a PriceSeries directly
prices = fy.PriceSeries(100 * np.cumprod(1 + np.random.randn(750) * 0.01))

# 2. Compose a strategy: momentum feature -> position -> backtest with costs
strat = fy.Strategy(
    features=lambda p: np.sign(np.diff(p, prepend=p[0])),
    signal=lambda x: x,
    cost=fy.ProportionalCost(fee=0.0005),
)
result = strat.run(prices)

# 3. Evaluate and report
print(result.summary())     # Sharpe, Sortino, Calmar, max drawdown, ...
fig = fy.tearsheet(result)  # one-call performance report
```

See [`Notebooks/quickstart_v2.ipynb`](Notebooks/quickstart_v2.ipynb) for the full
runnable tour (data, features, walk-forward, reporting). An optional Streamlit
playground ships under [`apps/playground/`](apps/playground/)
(`pip install -e ".[ui]" && streamlit run apps/playground/app.py`).

## Links

- PyPI: https://pypi.org/project/fynance/
- Documentation: https://fynance.readthedocs.io/en/latest/
- Source: https://github.com/ArthurBernard/Fynance
- Changelog: https://github.com/ArthurBernard/Fynance/blob/master/CHANGELOG.md
