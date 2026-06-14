#!/usr/bin/env python3
# coding: utf-8

""" Result container for walk-forward cross-validation. """

# Built-in packages
from __future__ import annotations

from dataclasses import dataclass

# Third-party packages
import numpy as np

__all__ = ['CVResult']


@dataclass
class CVResult:
    """ Results from :meth:`_RollingBasis.cross_validate`.

    Attributes
    ----------
    oof_predictions : np.ndarray
        Out-of-fold predictions, shape ``(T, n_out)``.  Positions that
        fall before the first test fold are filled with ``NaN``.
    fold_metrics : list of float
        Per-fold metric values (empty list when ``metric_fn`` is None).
    mean_metric : float or None
        Mean of ``fold_metrics``, or None when no metric was provided.
    std_metric : float or None
        Standard deviation of ``fold_metrics``, or None when no metric
        was provided.

    """

    oof_predictions: np.ndarray
    fold_metrics: list
    mean_metric: float | None
    std_metric: float | None
