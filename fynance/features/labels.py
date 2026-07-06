#!/usr/bin/env python3
# coding: utf-8

""" AFML-style path-aware labeling: triple-barrier, meta-labels, weights.

.. warning::

   **These functions look at FUTURE prices BY DESIGN.** They build training
   *targets* (``y``), never features (``X``). A ``triple_barrier`` label at
   ``t_in`` is only known once the barrier is actually touched, at ``t_out >
   t_in``. Feeding any field of the output back into a model as an input at
   ``t_in`` (or at any time before ``t_out``) is lookahead bias. Always route
   these labels through a purged / embargoed split
   (:mod:`fynance.data.split` — :func:`~fynance.data.split.train_test_split`,
   :func:`~fynance.data.split.walk_forward`) before training, so that no
   training fold ends inside the ``[t_in, t_out]`` span of a label used for
   validation.

Implements the labeling scheme from Lopez de Prado, *Advances in Financial
Machine Learning* (Wiley, 2018), chapters 3-4:

- :func:`triple_barrier` — path-dependent labels from two horizontal barriers
  (profit-take / stop-loss, in volatility-scaled simple-return units) and one
  vertical barrier (a maximum holding period).
- :func:`meta_labels` — binarizes a *primary* model's side prediction against
  the realized :func:`triple_barrier` outcome, for a secondary
  ("size the bet") model.
- :func:`label_concurrency` and :func:`uniqueness_weights` — because
  overlapping labels share the same underlying price path, they are not i.i.d.
  observations; :func:`uniqueness_weights` down-weights samples in proportion
  to how much they overlap other labels.

"""

from __future__ import annotations

# Third-party packages
import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = [
    'triple_barrier', 'meta_labels', 'label_concurrency', 'uniqueness_weights',
]

#: Structured dtype of the array returned by :func:`triple_barrier`.
LABEL_DTYPE = np.dtype([
    ('t_in', np.int64), ('t_out', np.int64), ('label', np.int8),
    ('ret', np.float64),
])


# --------------------------------------------------------------------------- #
#   numba kernel                                                              #
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _triple_barrier_kernel(prices, events, pt, sl, max_holding, scale):
    """ Walk each event forward until a barrier is touched or time runs out. """
    n = events.shape[0]
    T = prices.shape[0]
    t_in = np.empty(n, dtype=np.int64)
    t_out = np.empty(n, dtype=np.int64)
    label = np.empty(n, dtype=np.int8)
    ret = np.empty(n, dtype=np.float64)

    for k in range(n):
        i = events[k]
        p0 = prices[i]
        upper = pt * scale[k]
        lower = -sl * scale[k]

        vertical = i + max_holding
        if vertical > T - 1:
            vertical = T - 1

        lab = np.int8(0)
        out_j = vertical
        for j in range(i + 1, vertical + 1):
            r = prices[j] / p0 - 1.0
            if r >= upper:
                lab = np.int8(1)
                out_j = j
                break
            if r <= lower:
                lab = np.int8(-1)
                out_j = j
                break

        t_in[k] = i
        t_out[k] = out_j
        label[k] = lab
        ret[k] = prices[out_j] / p0 - 1.0

    return t_in, t_out, label, ret


# --------------------------------------------------------------------------- #
#   public API                                                                 #
# --------------------------------------------------------------------------- #


def triple_barrier(
    prices: NDArray,
    events: NDArray | None = None,
    pt: float = 1.0,
    sl: float = 1.0,
    max_holding: int = 21,
    vol: NDArray | None = None,
) -> NDArray:
    r""" Path-dependent triple-barrier labels (AFML ch. 3).

    .. warning::

       This is a **label**, not a feature — it is only known at ``t_out``,
       strictly after ``t_in``. Never use it as a model input; see the module
       warning and route it through :mod:`fynance.data.split`.

    From each event start ``i`` (``t_in``), two horizontal barriers are set on
    the **simple return** of ``prices`` relative to ``prices[i]``: an upper
    barrier at ``+pt * scale_i`` and a lower barrier at ``-sl * scale_i``. The
    path is then walked forward one bar at a time over ``(i, i + max_holding]``
    (clipped to the last index): the first bar that touches a barrier sets the
    label (``+1`` upper, ``-1`` lower); if neither is touched before the
    vertical barrier, the label is ``0`` at that vertical bar.

    .. math::

        \tau_i = \min\{t > i : r_{i,t} \geq pt \cdot scale_i
                              \text{ or } r_{i,t} \leq -sl \cdot scale_i\}
                 \wedge (i + max\_holding)

    with :math:`r_{i,t} = prices_t / prices_i - 1`.

    Parameters
    ----------
    prices : np.ndarray[dtype, ndim=1]
        One-dimensional series of price levels, length ``T``.
    events : np.ndarray[int], optional
        Integer indices of label start times (``t_in``). Must lie in
        ``[0, T - 2]`` so that at least one future bar exists. Default is
        ``np.arange(T - 1)`` (one label per bar, except the last).
    pt, sl : float, optional
        Profit-take / stop-loss barrier width, as a multiple of ``scale_i``.
        Both must be strictly positive. Default ``1.0`` for both.
    max_holding : int, optional
        Maximum holding period (vertical barrier), in bars. Must be ``>= 1``.
        Default 21.
    vol : np.ndarray[dtype, ndim=1], optional
        Per-bar, **return-scale** volatility, shape ``(T,)`` (e.g. a causal
        trailing realized volatility of simple returns — see
        :func:`~fynance.features.indicators.realized_volatility` or
        :func:`~fynance.features.momentums.smstd` on returns). ``scale_i`` is
        ``vol[i]``. If ``None`` (default), ``scale_i`` is the **constant**
        standard deviation of the 1-bar simple returns computed over the
        *whole* series — this is an in-sample quantity (it peeks at the
        entire history, future included) kept only as a convenience default;
        pass a causal ``vol`` for a genuinely leakage-free label.

    Returns
    -------
    np.ndarray[LABEL_DTYPE, ndim=1]
        Structured array of length ``len(events)`` with fields:

        - ``t_in`` (``int64``) — label start index (== ``events``).
        - ``t_out`` (``int64``) — index where the label was resolved.
        - ``label`` (``int8``) — ``+1`` upper barrier, ``-1`` lower barrier,
          ``0`` vertical barrier (timeout).
        - ``ret`` (``float64``) — simple return from ``t_in`` to ``t_out``.

    Raises
    ------
    ValueError
        If ``prices`` has fewer than 2 observations; if any ``events`` index
        is outside ``[0, T - 2]``; if ``pt <= 0`` or ``sl <= 0``; if
        ``max_holding < 1``; or if ``vol`` is given with a shape other than
        ``(T,)``.

    Notes
    -----
    Barrier checks favor the upper barrier on a tie (both touched on the same
    bar, which can only happen with ``scale_i = 0``).

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([100., 101., 105., 101., 98., 97.])
    >>> vol = np.full(6, 0.02)
    >>> out = triple_barrier(prices, events=np.array([0]), max_holding=4, vol=vol)
    >>> out['t_in'], out['t_out'], out['label'], out['ret']
    (array([0]), array([2]), array([1], dtype=int8), array([0.05]))

    A path that never touches either barrier resolves at the vertical bar:

    >>> flat = np.array([100., 100.5, 100.2, 100.8, 100.1])
    >>> out = triple_barrier(flat, events=np.array([0]), max_holding=3, vol=np.full(5, 1.0))
    >>> out['t_out'], out['label']
    (array([3]), array([0], dtype=int8))

    See Also
    --------
    meta_labels, label_concurrency, uniqueness_weights
    fynance.data.split.train_test_split, fynance.data.split.walk_forward

    References
    ----------
    .. [1] Lopez de Prado, M. (2018). *Advances in Financial Machine
       Learning*. Wiley. Chapter 3, "Labeling".

    """
    prices = np.asarray(prices, dtype=np.float64).reshape(-1)
    T = prices.shape[0]

    if T < 2:

        raise ValueError(f"prices must have at least 2 observations, got {T}")

    if events is None:
        events = np.arange(T - 1, dtype=np.int64)
    else:
        events = np.asarray(events, dtype=np.int64).reshape(-1)
        if events.size and (events.min() < 0 or events.max() > T - 2):

            raise ValueError(
                f"event indices must be in [0, {T - 2}] to leave at least one "
                f"future bar for the label, got range [{events.min()}, "
                f"{events.max()}]"
            )

    if pt <= 0:

        raise ValueError(f"pt must be strictly positive, got {pt!r}")

    if sl <= 0:

        raise ValueError(f"sl must be strictly positive, got {sl!r}")

    max_holding = int(max_holding)
    if max_holding < 1:

        raise ValueError(f"max_holding must be >= 1, got {max_holding}")

    if vol is None:
        simple_ret = prices[1:] / prices[:-1] - 1.0
        scale = np.full(events.shape[0], simple_ret.std(), dtype=np.float64)
    else:
        vol = np.asarray(vol, dtype=np.float64).reshape(-1)
        if vol.shape != (T,):

            raise ValueError(
                f"vol must have shape ({T},) matching prices, got {vol.shape}"
            )

        scale = vol[events]

    t_in, t_out, label, ret = _triple_barrier_kernel(
        prices, events, float(pt), float(sl), max_holding, scale,
    )

    out = np.empty(events.shape[0], dtype=LABEL_DTYPE)
    out['t_in'] = t_in
    out['t_out'] = t_out
    out['label'] = label
    out['ret'] = ret

    return out


def meta_labels(side: NDArray, labels: NDArray) -> NDArray:
    r""" Meta-label a primary side prediction against realized outcomes.

    .. warning::

       This is a **label**, not a feature — see the module warning.

    A meta-label answers "was the primary model's side call correct?"; it is
    the target of a *secondary* model that learns to size (or filter) the
    primary model's bets (AFML ch. 3).

    Parameters
    ----------
    side : np.ndarray[dtype, ndim=1]
        Primary model's predicted direction per event, in ``{-1, 0, +1}`` (0
        meaning "no bet"), aligned one-to-one with ``labels``.
    labels : np.ndarray[LABEL_DTYPE, ndim=1]
        Output of :func:`triple_barrier` (only the ``ret`` field is used).

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        ``1.0`` where ``side * ret > 0`` (the side call and the realized
        return agree), ``0.0`` otherwise — including every ``side == 0`` bet
        and every ``ret == 0`` (vertical-barrier) outcome.

    Raises
    ------
    ValueError
        If ``side`` and ``labels`` have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array(
    ...     [(0, 1, 1, 0.05), (0, 2, -1, -0.03), (0, 3, 0, 0.0)],
    ...     dtype=LABEL_DTYPE,
    ... )
    >>> meta_labels(np.array([1, 1, -1]), labels)
    array([1., 0., 0.])

    See Also
    --------
    triple_barrier

    References
    ----------
    .. [1] Lopez de Prado, M. (2018). *Advances in Financial Machine
       Learning*. Wiley. Chapter 3, "Labeling".

    """
    side = np.asarray(side, dtype=np.float64).reshape(-1)
    ret = np.asarray(labels['ret'], dtype=np.float64)

    if side.shape[0] != ret.shape[0]:

        raise ValueError(
            f"side and labels must align, got {side.shape[0]} vs {ret.shape[0]}"
        )

    return (side * ret > 0.0).astype(np.float64)


def label_concurrency(t_in: NDArray, t_out: NDArray, T: int) -> NDArray:
    r""" Number of labels alive at each bar (inclusive ``[t_in, t_out]`` spans).

    .. warning::

       This describes **labels**, not a feature — see the module warning.

    Used by :func:`uniqueness_weights`, and useful on its own to diagnose how
    much :func:`triple_barrier` labels overlap.

    Parameters
    ----------
    t_in, t_out : np.ndarray[int, ndim=1]
        Label start / end indices (the ``t_in`` / ``t_out`` fields of a
        :func:`triple_barrier` output), same length, with
        ``0 <= t_in <= t_out <= T - 1`` element-wise.
    T : int
        Number of bars in the underlying series.

    Returns
    -------
    np.ndarray[np.int64, ndim=1]
        Shape ``(T,)``, the count of labels whose ``[t_in, t_out]`` span
        includes each bar.

    Raises
    ------
    ValueError
        If ``t_in`` and ``t_out`` have different lengths, or any span falls
        outside ``[0, T - 1]`` or has ``t_in > t_out``.

    Examples
    --------
    >>> import numpy as np
    >>> label_concurrency(np.array([0, 0, 2]), np.array([1, 1, 3]), T=4)
    array([2, 2, 1, 1])

    See Also
    --------
    uniqueness_weights, triple_barrier

    References
    ----------
    .. [1] Lopez de Prado, M. (2018). *Advances in Financial Machine
       Learning*. Wiley. Chapter 4, "Sample Weights".

    """
    t_in = np.asarray(t_in, dtype=np.int64).reshape(-1)
    t_out = np.asarray(t_out, dtype=np.int64).reshape(-1)

    if t_in.shape[0] != t_out.shape[0]:

        raise ValueError("t_in and t_out must have the same length")

    if t_in.shape[0] and (
        np.any(t_in < 0) or np.any(t_out > T - 1) or np.any(t_in > t_out)
    ):

        raise ValueError("expected 0 <= t_in <= t_out <= T - 1 for every label")

    diff = np.zeros(T + 1, dtype=np.int64)
    np.add.at(diff, t_in, 1)
    np.add.at(diff, t_out + 1, -1)

    return np.cumsum(diff)[:T]


def uniqueness_weights(t_in: NDArray, t_out: NDArray, T: int) -> NDArray:
    r""" Average-uniqueness sample weights for overlapping labels (AFML ch. 4).

    .. warning::

       This produces sample **weights** derived from labels, not a feature —
       see the module warning.

    Overlapping :func:`triple_barrier` labels are drawn from overlapping price
    paths, so they are not i.i.d.: a bar covered by many concurrent labels
    gets diluted "credit" in each of them. For each label, the uniqueness at a
    bar is ``1 / concurrency`` there (see :func:`label_concurrency`); the
    weight is the label's *average* uniqueness over its own
    ``[t_in, t_out]`` span, rescaled so the weights sum to ``n`` (the number
    of labels) — the AFML convention, so the mean weight is 1 regardless of
    how much overlap there is.

    .. math::

        \bar u_i = \frac{1}{t_{out,i} - t_{in,i} + 1}
                   \sum_{t=t_{in,i}}^{t_{out,i}} \frac{1}{c_t}, \qquad
        w_i = \bar u_i \cdot \frac{n}{\sum_j \bar u_j}

    where :math:`c_t` is :func:`label_concurrency`.

    This composes **multiplicatively** with a recency scheme such as
    :func:`fynance.models.training.exp_sample_weights`: multiply the two
    weight vectors together (both already convey "importance", one for
    overlap and one for recency) rather than choosing one over the other.

    Parameters
    ----------
    t_in, t_out : np.ndarray[int, ndim=1]
        Label start / end indices, as in :func:`label_concurrency`.
    T : int
        Number of bars in the underlying series.

    Returns
    -------
    np.ndarray[np.float64, ndim=1]
        Shape ``(n,)`` where ``n = len(t_in)``, non-negative, summing to
        ``n`` (up to floating-point error). Disjoint labels all get weight
        ``1.0``; more overlap means a smaller weight.

    Raises
    ------
    ValueError
        Propagated from :func:`label_concurrency` for malformed spans.

    Examples
    --------
    Disjoint labels share no overlap, so every weight is 1:

    >>> import numpy as np
    >>> uniqueness_weights(np.array([0, 2]), np.array([1, 3]), T=4)
    array([1., 1.])

    Two identical, fully overlapping labels split their combined uniqueness
    evenly, and a third, disjoint label keeps its full weight -- rescaled so
    all three sum to 3:

    >>> uniqueness_weights(np.array([0, 0, 2]), np.array([1, 1, 3]), T=4)
    array([0.75, 0.75, 1.5 ])

    See Also
    --------
    label_concurrency, triple_barrier
    fynance.models.training.exp_sample_weights

    References
    ----------
    .. [1] Lopez de Prado, M. (2018). *Advances in Financial Machine
       Learning*. Wiley. Chapter 4, "Sample Weights".

    """
    t_in = np.asarray(t_in, dtype=np.int64).reshape(-1)
    t_out = np.asarray(t_out, dtype=np.int64).reshape(-1)
    n = t_in.shape[0]

    conc = label_concurrency(t_in, t_out, T)

    u = np.empty(n, dtype=np.float64)
    for k in range(n):
        u[k] = np.mean(1.0 / conc[t_in[k]:t_out[k] + 1])

    if n == 0:

        return u

    total = u.sum()
    if total <= 0:

        raise ValueError("uniqueness weights sum to zero or less (degenerate input)")

    return u * (n / total)


if __name__ == '__main__':

    import doctest

    doctest.testmod()
