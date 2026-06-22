#!/usr/bin/env python3
# coding: utf-8

""" Strictly time-ordered splits for ML evaluation.

No shuffling, ever. Provides a simple train/test split with an optional embargo
and a purged walk-forward window generator. These are pure index generators,
decoupled from any model (mirroring the walk-forward semantics of
:class:`fynance.models.rolling._RollingBasis`).

"""

from __future__ import annotations

# Built-in packages
from typing import Iterator

# Third-party packages
import numpy as np
from numpy.typing import NDArray

__all__ = ['train_test_split', 'walk_forward']


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
