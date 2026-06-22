#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Synthetic price generators.

Seeded synthetic price paths so the whole research harness is testable with
**zero real data**. They also serve as a **null test**: a strategy should show
~no skill on data with no real edge, which is a quick sanity check on the
guardrails. Real-data adapters live downstream (a private research repo), never
here.

"""

# Built-in
from __future__ import annotations

# Third-party
import numpy as np
from numpy.typing import NDArray

# Local
from fynance.core import PriceSeries

__all__ = ['gbm', 'regime_switching']


def gbm(
    n: int,
    *,
    mu: float = 0.0,
    sigma: float = 0.01,
    s0: float = 100.0,
    seed: int | None = None,
) -> PriceSeries:
    """ Geometric Brownian motion price path.

    Log-returns are drawn i.i.d. ``Normal(mu, sigma)``, so the path's mean
    log-return is ``mu`` and its volatility ``sigma``.

    Parameters
    ----------
    n : int
        Number of observations (path length).
    mu : float
        Mean per-step log-return.
    sigma : float
        Per-step log-return volatility.
    s0 : float
        Initial price.
    seed : int, optional
        Seed for reproducibility. ``None`` is nondeterministic.

    Returns
    -------
    fynance.core.PriceSeries
        Price path of length ``n``.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.research import gbm
    >>> a, b = gbm(5, seed=7), gbm(5, seed=7)
    >>> bool(np.allclose(a.to_numpy(), b.to_numpy()))
    True
    >>> int(a.to_numpy().size)
    5

    """
    rng = np.random.default_rng(seed)
    log_ret = mu + sigma * rng.standard_normal(max(n - 1, 0))
    path = np.concatenate([[0.0], np.cumsum(log_ret)])

    return PriceSeries(s0 * np.exp(path), name="synthetic-gbm")


def regime_switching(
    n: int,
    *,
    regimes: tuple[tuple[float, float], ...] = ((0.0, 0.01), (0.0, 0.03)),
    p_switch: float = 0.02,
    s0: float = 100.0,
    seed: int | None = None,
) -> PriceSeries:
    """ Markov regime-switching price path.

    The initial regime is drawn uniformly (rather than always starting in
    regime 0, which biased short paths toward the first regime). At each
    subsequent step the active regime switches (to a uniformly-drawn regime)
    with probability ``p_switch``; log-returns are then drawn from the active
    regime's ``(mu, sigma)``. The varying volatility makes it a natural input
    for :func:`fynance.detect_regimes`.

    Parameters
    ----------
    n : int
        Number of observations (path length).
    regimes : tuple of (float, float)
        ``(mu, sigma)`` per regime.
    p_switch : float
        Per-step probability of switching regime.
    s0 : float
        Initial price.
    seed : int, optional
        Seed for reproducibility. ``None`` is nondeterministic.

    Returns
    -------
    fynance.core.PriceSeries
        Price path of length ``n``.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.research import regime_switching
    >>> a, b = regime_switching(5, seed=3), regime_switching(5, seed=3)
    >>> bool(np.allclose(a.to_numpy(), b.to_numpy()))
    True

    """
    rng = np.random.default_rng(seed)
    mus = np.array([r[0] for r in regimes], dtype=np.float64)
    sigmas = np.array([r[1] for r in regimes], dtype=np.float64)
    k = mus.size

    steps = max(n - 1, 0)
    states: NDArray[np.int_] = np.empty(steps, dtype=np.int_)
    # Draw the initial regime uniformly so short paths are not biased toward
    # regime 0; subsequent steps switch with probability ``p_switch``.
    state = int(rng.integers(0, k))
    for t in range(steps):
        if rng.random() < p_switch:
            state = int(rng.integers(0, k))
        states[t] = state

    log_ret = mus[states] + sigmas[states] * rng.standard_normal(steps)
    path = np.concatenate([[0.0], np.cumsum(log_ret)])

    return PriceSeries(s0 * np.exp(path), name="synthetic-regime")
