#!/usr/bin/env python3
# coding: utf-8

""" Stacked direction + magnitude ensemble with a meta-model.

Two base models — one specialized in the *direction* of the move, one in its
*magnitude* — are combined by a meta-model trained on their **out-of-fold**
predictions (walk-forward OOF), which keeps the meta-features leak-free.
"""

from __future__ import annotations

# Built-in packages
from typing import Callable

# Third-party packages
import numpy as np
import torch
from numpy.typing import NDArray

# Local packages
from fynance.models.rolling import _RollingBasis

__all__ = ['StackingEnsemble']


class StackingEnsemble:
    r""" Direction + magnitude stacking with an out-of-fold meta-model.

    The two base models are evaluated with walk-forward cross-validation; their
    out-of-fold (OOF) predictions become the meta-features on which the
    meta-model is trained (e.g. with :class:`fynance.models.loss.SharpeLoss`).
    Using OOF predictions avoids feeding the meta-model with in-sample base
    predictions, the classic stacking leakage.

    Parameters
    ----------
    direction_factory, magnitude_factory : callable
        No-arg callables returning base models (the
        :class:`~fynance.models._base.BaseNeuralNet` interface). Typically the
        direction model is trained with
        :class:`~fynance.models.loss.DirectionalAccuracyLoss`, the magnitude
        model with MSE or :class:`~fynance.models.loss.SortinoLoss`.
    meta_factory : callable
        Callable ``meta_factory(n_features) -> model`` returning the meta-model
        sized for the stacked features.

    """

    def __init__(
        self,
        direction_factory: Callable,
        magnitude_factory: Callable,
        meta_factory: Callable,
    ):
        self.direction_factory = direction_factory
        self.magnitude_factory = magnitude_factory
        self.meta_factory = meta_factory

    def fit_predict(
        self, X, y, train_period: int, test_period: int, roll_period: int,
        epochs: int = 1,
    ) -> NDArray:
        """ Fit the ensemble and return meta out-of-fold predictions.

        Parameters
        ----------
        X, y : torch.Tensor
            Input and target, shapes ``(T, N)`` and ``(T, M)``.
        train_period, test_period, roll_period : int
            Walk-forward window parameters.
        epochs : int, optional
            Training passes per fold and for the meta-model. Default 1.

        Returns
        -------
        np.ndarray
            Meta predictions of shape ``(T, M)``; rows before the first test
            fold (no OOF base features) are ``NaN``.

        """
        n_out = y.shape[1] if y.ndim > 1 else 1

        rb = _RollingBasis(X, y)
        rb(train_period=train_period, test_period=test_period,
           roll_period=roll_period)
        dir_oof = rb.cross_validate(
            self.direction_factory, X, y, epochs=epochs).oof_predictions
        mag_oof = rb.cross_validate(
            self.magnitude_factory, X, y, epochs=epochs).oof_predictions

        meta_x = np.hstack([dir_oof, mag_oof])
        mask = ~np.isnan(meta_x).any(axis=1)

        y_np = y.numpy() if hasattr(y, 'numpy') else np.asarray(y)
        x_meta = torch.from_numpy(meta_x[mask].astype(np.float32))
        y_meta = torch.from_numpy(y_np[mask].astype(np.float32))

        meta = self.meta_factory(meta_x.shape[1])
        for _ in range(epochs):
            meta.train_on(x_meta, y_meta)

        pred = meta.predict(x_meta)
        pred = pred.numpy() if hasattr(pred, 'numpy') else np.asarray(pred)

        out = np.full((X.shape[0], n_out), np.nan)
        out[mask] = pred.reshape(-1, n_out)

        return out
