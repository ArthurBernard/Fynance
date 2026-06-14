#!/usr/bin/env python3
# coding: utf-8

""" Robust training utilities: sample weighting and early stopping. """

from __future__ import annotations

# Third-party packages
import numpy as np
from numpy.typing import NDArray

__all__ = ['EarlyStopping', 'exp_sample_weights']


def exp_sample_weights(n: int, halflife: float) -> NDArray:
    r""" Exponential-decay sample weights (recent observations weighted more).

    The most recent observation gets weight 1; weights halve every
    ``halflife`` observations into the past. Useful to down-weight stale
    data when training on a long history.

    Parameters
    ----------
    n : int
        Number of observations.
    halflife : float
        Decay half-life in observations (must be > 0).

    Returns
    -------
    np.ndarray
        Weights of shape ``(n,)``, in ``(0, 1]``, increasing with index
        (oldest first, most recent last).

    Examples
    --------
    >>> exp_sample_weights(4, halflife=1)
    array([0.125, 0.25 , 0.5  , 1.   ])

    """
    if halflife <= 0:
        raise ValueError("halflife must be > 0")
    age = (n - 1) - np.arange(n)

    return 0.5 ** (age / halflife)


class EarlyStopping:
    """ Patience-based early stopping on a monitored metric.

    Call :meth:`step` with the validation metric each epoch; it returns
    ``True`` once the metric has failed to improve for ``patience`` steps.

    Parameters
    ----------
    patience : int, optional
        Number of non-improving steps to tolerate. Default 5.
    min_delta : float, optional
        Minimum change counted as an improvement. Default 0.
    mode : {'max', 'min'}, optional
        Whether the metric should be maximized (e.g. Sharpe) or minimized
        (e.g. loss). Default 'max'.

    Attributes
    ----------
    best : float or None
        Best metric seen so far.
    stop : bool
        Whether stopping has been triggered.

    """

    def __init__(self, patience: int = 5, min_delta: float = 0.,
                 mode: str = 'max'):
        if mode not in ('max', 'min'):
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.count = 0
        self.stop = False

    def step(self, metric: float) -> bool:
        """ Update with a new metric value; return True if training should stop. """
        if self.best is None:
            improved = True
        elif self.mode == 'max':
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

        return self.stop
