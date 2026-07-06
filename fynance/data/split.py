#!/usr/bin/env python3
# coding: utf-8

""" Strictly time-ordered splits for ML evaluation.

No shuffling, ever. Provides a simple train/test split with an optional embargo,
a purged walk-forward window generator, and a combinatorial purged
cross-validation (CPCV) splitter that yields many purged/embargoed train/test
folds instead of a single path. These are pure index generators, decoupled from
any model (mirroring the walk-forward semantics of
:class:`fynance.models.rolling._RollingBasis`).

"""

from __future__ import annotations

# Built-in packages
import itertools
from typing import Iterator

# Third-party packages
import numpy as np
from numpy.typing import NDArray

__all__ = ['train_test_split', 'walk_forward', 'combinatorial_purged_cv']


def train_test_split(
    n: int,
    test_size: float | int,
    gap: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """ Time-ordered train/test index split.

    Parameters
    ----------
    n : int
        Number of observations.
    test_size : float or int
        Trailing test set size. A value strictly inside ``(0, 1)`` is read as a
        **fraction** of ``n`` (e.g. ``0.2`` -> ``round(0.2 * n)``); any other
        value -- including the bounds ``0.0`` and ``1.0`` -- is read as an
        **absolute count** (``int(test_size)``). In particular ``1.0`` means a
        single observation (count ``1``), not the whole series, and ``0.0``
        means an empty test set; pass a fraction strictly between the bounds to
        get a proportional split.
    gap : int
        Embargo: observations dropped between train end and test start.

    Returns
    -------
    (train_idx, test_idx) : tuple of numpy.ndarray
        ``test_idx`` is strictly after ``train_idx`` (no leakage).

    Raises
    ------
    ValueError
        If ``test_size`` is negative (a negative integer would yield
        out-of-bounds train indices and a negative fraction would silently
        produce an empty test set), if the resulting test count exceeds ``n``,
        or if the train set would be empty.

    """
    if test_size < 0:

        raise ValueError(f"test_size must be >= 0, got {test_size}")

    n_test = int(round(n * test_size)) if 0 < test_size < 1 else int(test_size)

    if n_test > n:

        raise ValueError(f"test_size ({n_test}) exceeds n ({n})")

    split = n - n_test

    if split - gap <= 0:

        raise ValueError("train set is empty; reduce test_size/gap")

    train_idx = np.arange(0, split - gap, dtype=np.int64)
    test_idx = np.arange(split, n, dtype=np.int64)

    return train_idx, test_idx


def walk_forward(
    n: int,
    train: int,
    test: int,
    step: int | None = None,
    purge: int = 0,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """ Generate purged walk-forward windows.

    Each window trains on ``[t-train : t-purge]`` and tests on ``[t : t+test]``.

    Parameters
    ----------
    n : int
        Number of observations.
    train, test : int
        Train and test window lengths.
    step : int, optional
        Roll step (defaults to ``test``, i.e. non-overlapping test windows).
    purge : int
        Observations removed at the train/test boundary (embargo).

    Yields
    ------
    (train_idx, test_idx) : tuple of numpy.ndarray
        Index arrays with ``test_idx`` strictly after ``train_idx``.

    Raises
    ------
    ValueError
        If ``train <= 0`` or ``purge >= train``: either would yield empty train
        windows (``[t-train : t-purge]`` becomes empty), which silently breaks a
        downstream ``fit`` with an opaque error instead of failing here. Also if
        ``step <= 0``, which would never advance ``t`` and loop forever.

    """
    if train <= 0:

        raise ValueError(f"train must be > 0, got {train}")

    if purge >= train:

        raise ValueError(
            f"purge must be < train, got purge={purge}, train={train} "
            "(otherwise every train window is empty)"
        )

    if step is None:
        step = test

    if step <= 0:

        raise ValueError(
            f"step must be > 0, got {step} (otherwise t never advances "
            "and the window generator loops forever)"
        )

    t = train

    while t + test <= n:
        train_idx = np.arange(max(0, t - train), t - purge, dtype=np.int64)
        test_idx = np.arange(t, t + test, dtype=np.int64)

        yield train_idx, test_idx

        t += step


def combinatorial_purged_cv(
    T: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge: int = 0,
    embargo: int = 0,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """ Generate combinatorial purged cross-validation (CPCV) folds.

    A single walk-forward path (or a plain purged K-fold) gives exactly one
    out-of-sample (OOS) performance estimate, so its variance from the
    particular split chosen is never measured -- a lucky or unlucky path looks
    identical to a robust one. CPCV instead splits ``[0, T)`` into
    ``n_groups`` contiguous groups and, for **every** combination of
    ``n_test_groups`` groups held out as test, purges/embargoes the remainder
    and yields it as one train/test fold. That produces
    ``n_splits = math.comb(n_groups, n_test_groups)`` folds -- versus the
    single path of :func:`walk_forward` -- whose paths can be reassembled into
    many distinct OOS return series, turning one backtest into a distribution
    of OOS Sharpe ratios (and letting one estimate the probability of backtest
    overfitting, PBO).

    Parameters
    ----------
    T : int
        Number of observations.
    n_groups : int, optional
        Number of contiguous groups to split ``[0, T)`` into (sizes as equal
        as possible; any remainder bars go to the first groups, i.e. the same
        convention as :func:`numpy.array_split`). Default 6.
    n_test_groups : int, optional
        Number of groups held out as test in each fold. Default 2.
    purge : int, optional
        Number of bars removed from train on **both sides** of every test
        group's boundary (immediately before its start and immediately after
        its end), so that no train observation straddles a test boundary.
        Default 0.
    embargo : int, optional
        Number of bars *additionally* removed from train immediately after
        each test group's end (beyond ``purge``), to absorb serial
        correlation that would otherwise leak test information forward into
        a subsequent train block. Default 0.

    Yields
    ------
    (train_idx, test_idx) : tuple of numpy.ndarray
        Sorted, unique ``int64`` index arrays, same convention as
        :func:`walk_forward`. ``test_idx`` is the union of the chosen test
        groups; ``train_idx`` is every other index minus the purge/embargo
        windows.

    Raises
    ------
    ValueError
        If ``n_test_groups`` is not strictly between ``0`` and ``n_groups``,
        or if ``n_groups`` exceeds ``T`` (a group would then be empty).

    Notes
    -----
    Combinations are generated in the deterministic order of
    :func:`itertools.combinations` applied to ``range(n_groups)``. By a
    standard counting argument, each of the ``n_groups`` groups appears in
    the test set of exactly ``math.comb(n_groups - 1, n_test_groups - 1)`` of
    the ``math.comb(n_groups, n_test_groups)`` folds.

    Examples
    --------
    >>> import math
    >>> folds = list(combinatorial_purged_cv(12, n_groups=4, n_test_groups=1, purge=1))
    >>> len(folds) == math.comb(4, 1)
    True
    >>> train_idx, test_idx = folds[0]
    >>> test_idx
    array([0, 1, 2])
    >>> train_idx  # bar 3 purged (1 bar after the test group's end)
    array([ 4,  5,  6,  7,  8,  9, 10, 11])

    See Also
    --------
    train_test_split, walk_forward

    References
    ----------
    .. [1] Lopez de Prado, M. (2018). *Advances in Financial Machine
       Learning*. Wiley. Chapter 7, "Cross-Validation in Finance".

    """
    if not (0 < n_test_groups < n_groups):

        raise ValueError(
            "n_test_groups must satisfy 0 < n_test_groups < n_groups, got "
            f"n_test_groups={n_test_groups}, n_groups={n_groups}"
        )

    if n_groups > T:

        raise ValueError(f"n_groups ({n_groups}) exceeds T ({T})")

    sizes = np.full(n_groups, T // n_groups, dtype=np.int64)
    sizes[: T % n_groups] += 1
    bounds = np.concatenate(([0], np.cumsum(sizes)))
    starts, ends = bounds[:-1], bounds[1:]

    for combo in itertools.combinations(range(n_groups), n_test_groups):
        test_mask = np.zeros(T, dtype=bool)
        excl_mask = np.zeros(T, dtype=bool)

        for g in combo:
            s, e = int(starts[g]), int(ends[g])
            test_mask[s:e] = True
            excl_mask[max(0, s - purge):s] = True
            excl_mask[e:min(T, e + purge)] = True
            excl_mask[e:min(T, e + embargo)] = True

        train_mask = ~test_mask & ~excl_mask
        train_idx = np.nonzero(train_mask)[0].astype(np.int64)
        test_idx = np.nonzero(test_mask)[0].astype(np.int64)

        yield train_idx, test_idx
