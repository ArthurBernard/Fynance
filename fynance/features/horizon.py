#!/usr/bin/env python3
# coding: utf-8

""" Forward horizon-return labels.

Helpers to turn a price series into forward (look-ahead) return labels over a
fixed horizon ``h``. These are the *targets* an Information Coefficient
(:func:`fynance.metrics.information_coefficient`) is computed against. The
non-overlapping variant (the default) is the safe choice for IC estimation:
overlapping H-bar labels share data and inflate the apparent correlation.

"""

from __future__ import annotations

# Built-in packages
# Third-party packages
import numpy as np
from numpy.typing import NDArray

# Local packages

__all__ = ['horizon_returns']


def horizon_returns(
    prices: NDArray,
    h: int,
    *,
    overlapping: bool = False,
) -> NDArray:
    r""" Forward return over a fixed horizon ``h``, strictly causal.

    The label at time ``t`` is the simple return realized over the next ``h``
    steps, :math:`prices_{t+h} / prices_t - 1`. It looks *forward* (the label
    at ``t`` is only known at ``t + h``), so it is a prediction *target*, not a
    feature — never feed it back into the model as an input at time ``t``.

    By default the labels are **non-overlapping**: one sample every ``h`` steps
    (``t = 0, h, 2h, ...``), so no two labels share any underlying price move.
    This matters for Information Coefficient estimation — overlapping H-bar
    labels are autocorrelated by construction and inflate the apparent IC. Set
    ``overlapping=True`` to get one label per bar (``t = 0, 1, 2, ...``) when
    that autocorrelation is acceptable.

    Notes
    -----
    With :math:`T` observations and horizon :math:`h`, the forward return is

    .. math::

        r^{(h)}_t = \frac{prices_{t+h}}{prices_t} - 1

    defined for :math:`t + h < T`. The non-overlapping output keeps
    :math:`t \in \{0, h, 2h, \dots\}` and has length
    :math:`\lfloor (T - 1) / h \rfloor`; the overlapping output keeps every
    :math:`t \in \{0, 1, \dots\}` and has length :math:`T - h`.

    Parameters
    ----------
    prices : np.ndarray[dtype, ndim=1 or 2]
        Time-series of prices indexed by time on the first axis. For a 2-D
        ``(T, N)`` panel each column is an independent series.
    h : int
        Forward horizon in number of steps, must be a positive integer.
    overlapping : bool, optional
        If ``False`` (default), keep one label every ``h`` steps
        (non-overlapping). If ``True``, keep one label per bar (overlapping).

    Returns
    -------
    np.ndarray[np.float64, ndim=1 or 2]
        Forward horizon returns. A 1-D input of length ``T`` yields a 1-D
        output; a 2-D ``(T, N)`` input yields a ``(K, N)`` output where ``K`` is
        the number of retained labels.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rate_of_return

    Examples
    --------
    Non-overlapping 2-step forward returns (one label every 2 bars):

    >>> import numpy as np
    >>> prices = np.array([100., 110., 121., 133.1, 146.41, 161.051])
    >>> horizon_returns(prices, 2)
    array([0.21, 0.21])

    Overlapping labels keep one per bar:

    >>> horizon_returns(prices, 2, overlapping=True)
    array([0.21, 0.21, 0.21, 0.21])

    See Also
    --------
    fynance.metrics.information_coefficient

    """
    if not isinstance(h, (int, np.integer)) or h <= 0:

        raise ValueError(f"h must be a positive integer, got {h!r}")

    prices = np.asarray(prices, dtype=np.float64)

    if prices.ndim not in (1, 2):

        raise ValueError(
            f"only 1-D and 2-D arrays are supported, got ndim={prices.ndim}"
        )

    T = prices.shape[0]

    if T <= h:

        raise ValueError(
            f"prices must have more than h={h} observations on axis 0, got {T}"
        )

    step = h if not overlapping else 1
    base_idx = np.arange(0, T - h, step)
    fwd = prices[base_idx + h] / prices[base_idx] - 1.

    return fwd
