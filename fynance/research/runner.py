#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" The experiment runner.

:func:`run_experiment` takes a strategy and a price series, runs a walk-forward
(or single), cost-aware, seeded backtest through the existing maillons, and
returns a populated :class:`~fynance.research.Experiment`. Results are written
only to a caller-provided ``output_dir`` — fynance never persists them itself.

"""

# Built-in
from __future__ import annotations

from pathlib import Path
from typing import Any

# Third-party
import numpy as np
from numpy.typing import NDArray

# Local
from fynance.core import PriceSeries
from fynance.research.experiment import Experiment

__all__ = ['run_experiment']


def _to_array(data: Any) -> NDArray[np.float64]:
    """ Coerce a PriceSeries / array-like to a 1-D float64 array. """
    if isinstance(data, PriceSeries):
        return data.to_numpy()

    return np.asarray(data, dtype=np.float64).reshape(-1)


def _seed_everything(seed: int) -> None:
    """ Seed numpy (and torch if present) for reproducibility. """
    np.random.seed(seed)
    try:  # torch is optional at call time
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run_experiment(
    strategy: Any,
    data: Any,
    *,
    name: str,
    X: NDArray | None = None,
    y: NDArray | None = None,
    walk_forward: dict[str, Any] | None = None,
    costs: Any = None,
    period: int = 252,
    seed: int = 0,
    code: str | None = None,
    output_dir: str | Path | None = None,
) -> Experiment:
    """ Run a strategy experiment and return a populated :class:`Experiment`.

    Parameters
    ----------
    strategy : fynance.strategy.Strategy
        The composed strategy (features -> model -> signal -> cost).
    data : PriceSeries or array-like
        Price series the strategy runs on (coerced to float64).
    name : str
        Experiment slug (also the output sub-directory).
    X : array-like, optional
        Precomputed feature matrix aligned with ``data`` (rows = time). Passed
        through to the strategy — the way to feed exogenous / regime / multi-venue
        features the price-only featurizer cannot build. Must be causal.
    y : array-like, optional
        Supervised target for a model-based strategy run via ``walk_forward``.
        Defaults to zeros (ignored by rule-based strategies).
    walk_forward : dict, optional
        Window params for :meth:`Strategy.run_walk_forward`
        (``train``/``test``/``step``/``purge``). ``None`` runs a single pass.
    costs : CostModel, optional
        Overrides the strategy's own cost model for this run when given.
    period : int
        Annualization factor for the metrics.
    seed : int
        Master seed (numpy + torch).
    code : str, optional
        Generated strategy source, stored verbatim on the experiment.
    output_dir : str or pathlib.Path, optional
        When given, the experiment is saved under ``<output_dir>/<name>/``.

    Returns
    -------
    Experiment
        Populated with ``spec``, ``metrics`` and ``series``.

    """
    _seed_everything(seed)

    if costs is not None:
        strategy.cost = costs

    prices = _to_array(data)
    n = prices.shape[0]

    if walk_forward is not None:
        target = np.zeros(n) if y is None else np.asarray(y)  # preserve dtype
        result = strategy.run_walk_forward(data, target, **walk_forward, X=X)

    else:
        result = strategy.run(data, y, X=X)

    metrics = result.summary(period=period)

    spec: dict[str, Any] = {
        "data": {"kind": getattr(data, "name", None) or type(data).__name__,
                 "n": int(n)},
        "features": None if X is None else {"X_shape": list(np.asarray(X).shape)},
        "walk_forward": walk_forward,
        "cost": type(strategy.cost).__name__ if strategy.cost is not None else None,
        "period": period,
        "seed": seed,
    }

    series = {
        "equity": np.asarray(result.equity, dtype=float).tolist(),
        "returns": np.asarray(result.returns, dtype=float).tolist(),
    }

    experiment = Experiment(
        name=name,
        spec=spec,
        code=code,
        seed=seed,
        metrics={k: float(v) for k, v in metrics.items()},
        series=series,
    )

    if output_dir is not None:
        experiment.save(output_dir, name=name)

    return experiment
