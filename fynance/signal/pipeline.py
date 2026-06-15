#!/usr/bin/env python3
# coding: utf-8

""" Compose a predictive model with a signal mapper. """

from __future__ import annotations

# Built-in packages
from typing import Any, Callable

# Third-party packages
from numpy.typing import NDArray

__all__ = ['SignalPipeline']


class SignalPipeline:
    """ A :class:`~fynance.core.protocols.SignalModel` plus a position mapper.

    Wraps a model exposing ``fit``/``predict`` and a mapper turning predictions
    into positions, exposing :meth:`predict_position`.

    Parameters
    ----------
    model : SignalModel
        Object with ``fit(X, y)`` and ``predict(X)``.
    mapper : callable
        Function mapping a prediction array to positions (e.g.
        :func:`fynance.signal.sign`).
    mapper_kwargs : dict, optional
        Extra keyword arguments forwarded to ``mapper``.

    """

    def __init__(
        self,
        model: Any,
        mapper: Callable[..., NDArray],
        mapper_kwargs: dict[str, Any] | None = None,
    ):
        """ Store the model and the mapper. """
        self.model = model
        self.mapper = mapper
        self.mapper_kwargs = mapper_kwargs or {}

    def fit(self, X: NDArray, y: NDArray) -> SignalPipeline:
        """ Fit the underlying model and return ``self``. """
        self.model.fit(X, y)

        return self

    def predict(self, X: NDArray) -> NDArray:
        """ Return the raw model predictions. """
        return self.model.predict(X)

    def predict_position(self, X: NDArray) -> NDArray:
        """ Predict, then map predictions to positions. """
        return self.mapper(self.model.predict(X), **self.mapper_kwargs)
