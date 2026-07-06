#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Combinatorially symmetric cross-validation (CSCV) overfitting guard.

Complements :mod:`fynance.research.guards`: where the permutation test and
the (deflated) Sharpe ratio ask *"is this one strategy's edge real?"*, the
**probability of backtest overfitting** (PBO) asks *"across everything I
tried, did I just pick the best-in-sample fluke?"*. It implements the CSCV
procedure of Bailey, Borwein, Lopez de Prado & Zhu, "The probability of
backtest overfitting", Journal of Computational Finance (2015): split the
track record into blocks, replay every balanced in-sample/out-of-sample split,
and measure how often the in-sample winner turns out to be an out-of-sample
laggard.

All functions are data-agnostic and pure numpy; nothing here reads real data
or stores results.

"""

# Built-in
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

# Third-party
import numpy as np
from numpy.typing import NDArray

# Local
from fynance.research.experiment import Experiment

__all__ = ['PBOResult', 'pbo', 'returns_panel']

# CSCV enumerates C(n_blocks, n_blocks / 2) IS/OOS splits; this is the value
# at the recommended n_blocks=16 ceiling (Bailey et al. run 16 blocks in their
# reference examples) — beyond it the combinatorics explode silently.
_MAX_COMBINATIONS = 12870


def returns_panel(experiments: list[Experiment]) -> NDArray[np.float64]:
    """ Build a ``(T, n_configs)`` returns panel from research experiments.

    One column per :class:`~fynance.research.Experiment`, taken from its
    stored ``series["returns"]`` (the per-period net returns
    :func:`~fynance.research.run_experiment` records). When an experiment
    carries only ``series["equity"]`` (e.g. a hand-built record), the column
    is derived as the equity curve's percentage change, ``R_t = E_t / E_{t-1}
    - 1`` for ``t = 1..T-1`` — one observation shorter than the equity curve,
    since the first equity point has no preceding value to return from.

    Parameters
    ----------
    experiments : list of Experiment
        The configurations to stack into a panel (>= 1).

    Returns
    -------
    numpy.ndarray, shape (T, n_configs)
        Per-period returns, one config per column.

    Raises
    ------
    ValueError
        If ``experiments`` is empty, an experiment carries neither a
        ``"returns"`` nor an ``"equity"`` series (or an equity curve too
        short to derive a return from), or the resulting columns have
        mismatched lengths.

    Examples
    --------
    >>> from fynance.research import Experiment, returns_panel
    >>> a = Experiment(name="a", series={"returns": [0.01, -0.02, 0.03]})
    >>> b = Experiment(name="b", series={"equity": [100.0, 101.0, 99.0, 102.0]})
    >>> returns_panel([a, b]).shape
    (3, 2)

    """
    if not experiments:
        raise ValueError("returns_panel needs at least one experiment")

    columns: list[NDArray[np.float64]] = []
    for exp in experiments:
        series = exp.series or {}
        if series.get("returns") is not None:
            col = np.asarray(series["returns"], dtype=np.float64)
        elif series.get("equity") is not None:
            equity = np.asarray(series["equity"], dtype=np.float64)
            if equity.shape[0] < 2:
                raise ValueError(
                    f"experiment {exp.name!r} equity curve is too short "
                    f"({equity.shape[0]} point(s)) to derive returns from"
                )
            col = equity[1:] / equity[:-1] - 1.0
        else:
            raise ValueError(
                f"experiment {exp.name!r} carries neither a 'returns' nor an "
                f"'equity' series"
            )
        columns.append(col)

    lengths = {col.shape[0] for col in columns}
    if len(lengths) > 1:
        details = {exp.name: col.shape[0] for exp, col in zip(experiments, columns)}
        raise ValueError(
            f"experiments have mismatched curve lengths, cannot form a panel: "
            f"{details}"
        )

    return np.column_stack(columns)


def _default_metric(returns: NDArray[np.float64]) -> float:
    """ Annualization-free Sharpe: mean over std of the per-period returns.

    Deliberately does not annualize by a ``period`` (CSCV only compares
    in-sample vs. out-of-sample *ranks* across configs, so a constant scale
    factor cancels) and does not assume ``returns`` is a price/equity curve
    the way :func:`fynance.metrics.sharpe` does — here it is already the
    per-period return series of one config's block-concatenated stretch.
    """
    if returns.size < 2:
        return 0.0

    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        # Degenerate (constant) stretch: no dispersion to divide by. Return a
        # large, finite, sign-carrying value rather than 0 or inf so a
        # genuinely constant-positive stretch still outranks noisy ones
        # without poisoning downstream arithmetic (OLS, rank ties) with inf.
        mean = float(np.mean(returns))
        return 0.0 if mean == 0.0 else math.copysign(1.0e6, mean)

    return float(np.mean(returns)) / std


def _rankdata_average(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Ascending, 1-indexed ranks with ties averaged (pure-numpy).

    A minimal analogue of ``scipy.stats.rankdata(x, method="average")`` —
    written locally to avoid adding a ``scipy.stats`` dependency for a single
    small helper on arrays sized ``n_configs``.
    """
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.shape[0], dtype=np.float64)
    n = x.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        # Ranks i+1..j+1 (1-indexed) tied -> their average.
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    return ranks


def _ols(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """ OLS slope/intercept of ``y`` on ``x``; flat line if ``x`` has no spread. """
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    var_x = float(np.sum((x - x_mean) ** 2))
    if var_x == 0.0:
        return 0.0, y_mean

    slope = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)

    return slope, y_mean - slope * x_mean


@dataclass
class PBOResult:
    """ Result of a :func:`pbo` run.

    Attributes
    ----------
    pbo : float
        Probability of backtest overfitting: the share of IS/OOS splits
        where the in-sample winner's out-of-sample relative rank falls at or
        below the median (``logit <= 0``).
    logits : numpy.ndarray, shape (n_combinations,)
        Rank logit ``ln(r / (1 - r))`` of the IS winner's OOS performance,
        one per split.
    prob_oos_loss : float
        Share of splits where the IS winner's OOS performance is negative.
    slope : float
        OLS slope of OOS performance on IS performance (of the IS winner)
        across splits — a robust-performing selection procedure has a
        positive slope; near-zero or negative indicates the IS ranking
        carries no out-of-sample information.
    intercept : float
        OLS intercept of the same regression.
    is_perf : numpy.ndarray, shape (n_combinations,)
        The IS winner's own in-sample metric value, one per split.
    oos_perf : numpy.ndarray, shape (n_combinations,)
        The IS winner's out-of-sample metric value, one per split.

    """

    pbo: float
    logits: NDArray[np.float64]
    prob_oos_loss: float
    slope: float
    intercept: float
    is_perf: NDArray[np.float64]
    oos_perf: NDArray[np.float64]


def pbo(
    returns_panel: NDArray[np.float64],
    n_blocks: int = 16,
    metric: Callable[[NDArray[np.float64]], float] | None = None,
) -> PBOResult:
    r""" Probability of backtest overfitting via combinatorially symmetric CV.

    Implements the CSCV procedure of Bailey, Borwein, Lopez de Prado & Zhu
    (2015): split the ``T`` periods into ``n_blocks`` contiguous blocks; for
    every way to assign half the blocks to an in-sample (IS) half and the
    other half to the complementary out-of-sample (OOS) half
    (:math:`\binom{n\_blocks}{n\_blocks / 2}` splits total), pick the config
    that scores best on the IS half by ``metric``, then locate that same
    config's OOS-half performance among *all* configs' OOS performances as a
    relative rank :math:`r \in (0, 1)` (average-tie rank divided by
    ``n_configs + 1``, so the best OOS config sits close to 1 and the worst
    close to 0). The rank logit is :math:`\lambda = \ln(r / (1 - r))`; ``pbo``
    is the share of splits with :math:`\lambda \le 0`, i.e. where the
    in-sample winner performs at or below the OOS median — a config search
    that overfits systematically sends the IS winner to the bottom half OOS.

    Parameters
    ----------
    returns_panel : numpy.ndarray, shape (T, n_configs)
        Per-period returns, one column per configuration tried (see
        :func:`returns_panel` to build this from
        :class:`~fynance.research.Experiment` runs).
    n_blocks : int
        Number of contiguous blocks to split ``T`` into. Must be even (a
        balanced IS/OOS split) and no greater than 16 — beyond that
        :math:`\binom{n\_blocks}{n\_blocks/2}` exceeds 12,870 splits.
    metric : callable, optional
        ``metric(returns) -> float`` scoring a 1-D per-period return stretch,
        higher is better. Defaults to an annualization-free Sharpe (mean over
        std of the returns) — deliberately not :func:`fynance.metrics.sharpe`,
        which assumes a price/equity curve rather than a returns series.

    Returns
    -------
    PBOResult
        The overfitting diagnostics (see :class:`PBOResult`).

    Raises
    ------
    ValueError
        If ``returns_panel`` is not 2-D, ``n_blocks`` is odd, ``n_blocks``
        implies more than 12,870 IS/OOS splits, or ``T < n_blocks``.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.research import pbo
    >>> rng = np.random.default_rng(0)
    >>> panel = rng.normal(0.0, 0.01, size=(400, 8))  # pure noise, no edge
    >>> result = pbo(panel, n_blocks=8)
    >>> 0.0 <= result.pbo <= 1.0
    True
    >>> result.logits.shape
    (70,)

    """
    panel = np.asarray(returns_panel, dtype=np.float64)
    if panel.ndim != 2:
        raise ValueError(
            f"returns_panel must be 2-D (T, n_configs); got ndim={panel.ndim}"
        )

    n_periods, n_configs = panel.shape
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even for a balanced split; got {n_blocks}")

    half = n_blocks // 2
    n_combinations = math.comb(n_blocks, half)
    if n_combinations > _MAX_COMBINATIONS:
        raise ValueError(
            f"n_blocks={n_blocks} implies C({n_blocks}, {half})={n_combinations} "
            f"IS/OOS splits (> {_MAX_COMBINATIONS}, i.e. n_blocks > 16) — lower "
            f"n_blocks"
        )

    if n_periods < n_blocks:
        raise ValueError(
            f"returns_panel has T={n_periods} periods, fewer than n_blocks="
            f"{n_blocks}"
        )

    fn = metric if metric is not None else _default_metric
    blocks = np.array_split(np.arange(n_periods), n_blocks)
    all_blocks = set(range(n_blocks))

    logits = np.empty(n_combinations, dtype=np.float64)
    is_perf = np.empty(n_combinations, dtype=np.float64)
    oos_perf = np.empty(n_combinations, dtype=np.float64)

    for k, is_combo in enumerate(combinations(range(n_blocks), half)):
        oos_combo = sorted(all_blocks - set(is_combo))
        is_idx = np.concatenate([blocks[b] for b in is_combo])
        oos_idx = np.concatenate([blocks[b] for b in oos_combo])

        is_metrics = np.array(
            [fn(panel[is_idx, c]) for c in range(n_configs)], dtype=np.float64
        )
        winner = int(np.argmax(is_metrics))

        oos_metrics = np.array(
            [fn(panel[oos_idx, c]) for c in range(n_configs)], dtype=np.float64
        )

        is_perf[k] = is_metrics[winner]
        oos_perf[k] = oos_metrics[winner]

        ranks = _rankdata_average(oos_metrics)
        r = ranks[winner] / (n_configs + 1)
        logits[k] = np.log(r / (1.0 - r))

    slope, intercept = _ols(is_perf, oos_perf)

    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        logits=logits,
        prob_oos_loss=float(np.mean(oos_perf < 0.0)),
        slope=slope,
        intercept=intercept,
        is_perf=is_perf,
        oos_perf=oos_perf,
    )
