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
        Fraction (``0 < x < 1``) or absolute count of the trailing test set.
    gap : int
        Embargo: observations dropped between train end and test start.

    Returns
    -------
    (train_idx, test_idx) : tuple of numpy.ndarray
        ``test_idx`` is strictly after ``train_idx`` (no leakage).

    """
    n_test = int(round(n * test_size)) if 0 < test_size < 1 else int(test_size)
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

    """
    if step is None:
        step = test

    t = train

    while t + test <= n:
        train_idx = np.arange(max(0, t - train), t - purge, dtype=np.int64)
        test_idx = np.arange(t, t + test, dtype=np.int64)

        yield train_idx, test_idx

        t += step
