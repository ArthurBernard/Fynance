#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Statistical guardrails against spurious results.

When an agent searches over many strategies, the danger is **overfitting and
data-snooping**: a good in-sample Sharpe may be luck or leakage. These guards make
honest evaluation cheap — a permutation test (is the edge real?) and the
probabilistic / deflated Sharpe ratio (does it survive the number of trials?).

All functions are data-agnostic and operate on the existing maillons / plain
metrics; nothing here reads real data or stores results.

"""

# Built-in
from __future__ import annotations

from typing import Any

# Third-party
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

# Local
from fynance.core import PriceSeries

__all__ = [
    'permutation_test',
    'probabilistic_sharpe_ratio',
    'deflated_sharpe_ratio',
]

# Euler-Mascheroni constant (for the expected maximum of N Gaussians).
_EULER = 0.5772156649015329


def _to_array(data: Any) -> NDArray[np.float64]:
    """ Coerce a PriceSeries / array-like to a 1-D float64 array. """
    if isinstance(data, PriceSeries):
        return data.to_numpy()

    return np.asarray(data, dtype=np.float64).reshape(-1)


def permutation_test(
    strategy: Any,
    data: Any,
    *,
    metric: str = "sharpe",
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """ Permutation test for a spurious edge.

    Runs the strategy on the real series, then on ``n_permutations`` shuffles of
    the asset's returns (which destroys any temporal structure). If the strategy
    scores as well on shuffled data as on the real data, its edge is not real.

    The price series must be strictly positive: the null is built on log-returns
    (``np.diff(np.log(prices))``), so a non-positive price would yield ``nan`` /
    ``-inf`` returns.

    Each run is seeded with a **distinct** seed derived from ``seed`` (the
    observed run and every permutation), so a stochastic strategy does not see
    the *same* model RNG draws on every path — which would bias the null
    variance. The whole test stays reproducible for a fixed ``seed``.

    Parameters
    ----------
    strategy : fynance.strategy.Strategy
        The strategy to evaluate.
    data : PriceSeries or array-like
        Strictly positive price series (``np.log`` is taken).
    metric : str
        Metric key from the run summary (default ``"sharpe"``).
    n_permutations : int
        Number of shuffles forming the null distribution.
    seed : int
        Master seed for the shuffles and the runs (per-run seeds are derived
        from it, so the test is fully reproducible).

    Returns
    -------
    dict
        ``observed``, ``p_value``, ``null_mean``, ``null_std``. The p-value is the
        (smoothed) fraction of shuffles scoring at least the observed metric.

    """
    # Imported here to avoid a runner <-> guards import cycle at module load.
    from fynance.research.runner import run_experiment

    prices = _to_array(data)
    log_ret = np.diff(np.log(prices))
    s0 = float(prices[0])

    # Draw distinct per-run seeds from the master seed so stochastic strategies
    # do not replay identical RNG draws on every path (which biases the null).
    seed_rng = np.random.default_rng(seed)
    run_seeds = seed_rng.integers(0, 2**31 - 1, size=n_permutations + 1)

    observed = run_experiment(strategy, prices, name="observed",
                              seed=int(run_seeds[0])).metrics[metric]

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        shuffled = rng.permutation(log_ret)
        path = s0 * np.exp(np.concatenate([[0.0], np.cumsum(shuffled)]))
        null[i] = run_experiment(strategy, path, name="perm",
                                 seed=int(run_seeds[i + 1])).metrics[metric]

    # Smoothed p-value (never exactly 0): (#{null >= observed} + 1) / (n + 1).
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))

    return {
        "observed": float(observed),
        "p_value": p_value,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
    }


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    *,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """ Probabilistic Sharpe ratio (PSR).

    The probability that the true Sharpe exceeds ``sr_benchmark`` given the
    estimate ``sr`` from ``n_obs`` observations, correcting for the returns'
    skewness and kurtosis (Bailey & López de Prado).

    Parameters
    ----------
    sr : float
        Observed (non-annualized) Sharpe ratio.
    n_obs : int
        Number of return observations.
    sr_benchmark : float
        Benchmark Sharpe to beat.
    skew : float
        Skewness of the returns.
    kurt : float
        Kurtosis of the returns (3 for a normal distribution).

    Returns
    -------
    float
        PSR in ``[0, 1]``.

    """
    if n_obs <= 1:
        return float("nan")

    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1) / denom

    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_variance: float = 1.0,
) -> float:
    """ Deflated Sharpe ratio (DSR).

    The PSR against a benchmark set to the **expected maximum** Sharpe of
    ``n_trials`` independent strategies — i.e. the probability the edge survives
    the multiple testing implied by trying ``n_trials`` strategies.

    Parameters
    ----------
    sr : float
        Observed **per-observation** (non-annualized) Sharpe ratio of the
        selected strategy. An annualized Sharpe must be divided by
        ``sqrt(period)`` first, or the DSR saturates to ~1.
    n_obs : int
        Number of return observations.
    n_trials : int
        Number of strategy configurations tried.
    skew, kurt : float
        Skewness / kurtosis of the selected strategy's returns.
    sr_variance : float
        Variance of the (per-observation) Sharpe estimates **across the
        trials**. The default ``1.0`` is a conservative placeholder — it
        overstates the expected-maximum benchmark and so understates the DSR;
        pass the empirical variance of the trial Sharpes whenever available
        (as :meth:`fynance.research.Ledger.deflated_sharpe` does).

    Returns
    -------
    float
        DSR in ``[0, 1]``. Low values flag a likely overfit selection.

    """
    n = max(int(n_trials), 1)
    if n == 1:
        sr_star = 0.0
    else:
        # Expected max of N i.i.d. standard normals, scaled by the SR dispersion.
        gauss_max = ((1 - _EULER) * norm.ppf(1 - 1.0 / n)
                     + _EULER * norm.ppf(1 - 1.0 / (n * np.e)))
        sr_star = float(np.sqrt(sr_variance) * gauss_max)

    return probabilistic_sharpe_ratio(
        sr, n_obs, sr_benchmark=sr_star, skew=skew, kurt=kurt
    )
