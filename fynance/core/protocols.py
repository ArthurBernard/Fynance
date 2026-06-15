#!/usr/bin/env python3
# coding: utf-8

""" Composition contracts for the fynance pipeline.

Structural :class:`typing.Protocol` seams the library composes through. They
are duck-typed (no forced inheritance): any object with the right shape
conforms. Only :class:`DataSource` is a *port* in the ports-&-adapters sense
(it touches the outside world); the others are internal composition seams.

Every protocol is numpy-typed and documents its causality contract: an output
at index ``t`` must depend only on inputs up to ``t`` (no lookahead).

"""

from __future__ import annotations

# Built-in packages
from typing import Any, Protocol, runtime_checkable

# Third-party packages
from numpy.typing import NDArray

__all__ = [
    'DataSource',
    'FeatureTransform',
    'SignalModel',
    'Allocator',
    'CostModel',
    'Metric',
]


@runtime_checkable
class DataSource(Protocol):
    """ Port: load external data into a :class:`PriceSeries`.

    The only I/O boundary protocol. Concrete adapters (CSV, Parquet, ...)
    live in :mod:`fynance.data`.
    """

    def load(self, *args: Any, **kwargs: Any) -> Any:
        """ Load and return a ``PriceSeries`` (or a mapping of them). """
        ...


@runtime_checkable
class FeatureTransform(Protocol):
    """ A stateful or stateless feature transformation.

    ``fit`` records only past-derivable parameters (e.g. train-slice mean and
    std); ``transform`` must be causal (output at ``t`` uses inputs up to
    ``t`` only).
    """

    def fit(self, X: NDArray) -> FeatureTransform:
        """ Fit any parameters and return ``self``. """
        ...

    def transform(self, X: NDArray) -> NDArray:
        """ Return the transformed array. """
        ...


@runtime_checkable
class SignalModel(Protocol):
    """ A predictive model mapping features to a target/signal.

    The numpy boundary of the model layer: ``predict`` returns numpy even when
    the implementation is pytorch internally.
    """

    def fit(self, X: NDArray, y: NDArray) -> SignalModel:
        """ Fit the model and return ``self``. """
        ...

    def predict(self, X: NDArray) -> NDArray:
        """ Return predictions as a numpy array. """
        ...


@runtime_checkable
class Allocator(Protocol):
    """ Map a covariance/return matrix to portfolio weights. """

    def __call__(self, data: NDArray) -> NDArray:
        """ Return an array of portfolio weights. """
        ...


@runtime_checkable
class CostModel(Protocol):
    """ Map a weight book to per-step transaction costs.

    Concrete models live in :mod:`fynance.backtest.cost`.
    """

    def __call__(self, weights: NDArray) -> NDArray:
        """ Return per-step costs aligned with the weight series. """
        ...


@runtime_checkable
class Metric(Protocol):
    """ Reduce a return series to a scalar performance number. """

    def __call__(self, returns: NDArray, *args: Any, **kwargs: Any) -> float:
        """ Return the scalar metric value. """
        ...
