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
    """ Coerce a PriceSeries / array-like to a float64 array.

    A 2-D ``(T, N)`` panel (multi-asset prices/returns) is preserved; anything
    else is flattened to a 1-D series.
    """
    if isinstance(data, PriceSeries):
        return data.to_numpy()

    arr = np.asarray(data, dtype=np.float64)

    return arr if arr.ndim == 2 else arr.reshape(-1)


def _index_bound(data: Any, pos: int) -> Any:
    """ Return a JSON-friendly index value at ``pos`` (first/-1), or None. """
    index = getattr(data, "index", None)
    if index is None:
        return None
    try:
        value = np.asarray(index).reshape(-1)[pos]
    except (IndexError, ValueError):
        return None
    # datetime64 -> ISO string; numpy scalars -> native python.
    if isinstance(value, np.datetime64):
        return str(value)

    return value.item() if hasattr(value, "item") else value


def _series_index(data: Any, n: int) -> list[str] | None:
    """ Tail-aligned datetime index (ISO strings) for an ``n``-point curve.

    Returns the last ``n`` index entries as ISO-8601 strings when ``data`` carries
    a genuine ``datetime64`` index long enough to cover the result curve; ``None``
    otherwise (the report then falls back to bar numbers). Right-aligned because a
    walk-forward equity curve covers the chronological *tail* of the series.
    """
    index = getattr(data, "index", None)
    if index is None:
        return None
    arr = np.asarray(index).reshape(-1)
    if not np.issubdtype(arr.dtype, np.datetime64) or arr.shape[0] < n:
        return None

    return [str(v) for v in arr[-n:]]


def _callable_name(fn: Any) -> str | None:
    """ Best-effort readable name for a signal/feature callable. """
    if fn is None:
        return None
    name = getattr(fn, "__name__", None)
    if name and name != "<lambda>":
        return name

    return type(fn).__name__ if not callable(fn) else name or "<callable>"


def _seed_everything(seed: int) -> None:
    """ Seed the global numpy (and torch) RNGs for reproducibility.

    This mutates process-global RNG state via :func:`numpy.random.seed` so that
    legacy strategies / models drawing from the global numpy RNG run
    deterministically. Callers that need *independent* runs (e.g.
    :func:`permutation_test`) must pass a **distinct** ``seed`` per run rather
    than reusing one — otherwise every run replays the same global draws.
    """
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
    feature_names: list[str] | None = None,
    feature_desc: str | None = None,
    data_desc: str | None = None,
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
    feature_names : list of str, optional
        Column names of ``X`` — recorded in the provenance so a report shows what
        each feature is. Ignored when ``X`` is None.
    feature_desc : str, optional
        Free-text description of how ``X`` was built (the feature recipe),
        recorded in the provenance. Ignored when ``X`` is None.
    data_desc : str, optional
        Free-text description of the price data (source, instrument, resolution),
        recorded in the provenance.
    output_dir : str or pathlib.Path, optional
        When given, the experiment is saved under ``<output_dir>/<name>/``.

    Notes
    -----
    The returned experiment's ``spec`` carries a self-describing **provenance**
    block (``data``, ``features``, ``model``, ``signal``, ``walk_forward``,
    ``cost``, ``period``, ``seed``) so an artifact always records *what produced
    it*. :func:`fynance.research.write_report` renders it as a Provenance table.

    Returns
    -------
    Experiment
        Populated with ``spec``, ``metrics`` and ``series``.

    """
    _seed_everything(seed)

    prices = _to_array(data)
    n = prices.shape[0]

    # Overriding the cost model must not leak past this run: save and restore
    # the strategy's own cost in a ``finally`` so a later run (or a sibling
    # iteration of a permutation loop sharing the strategy) is unaffected.
    _sentinel = object()
    original_cost: Any = getattr(strategy, "cost", _sentinel)
    try:
        if costs is not None:
            strategy.cost = costs

        if walk_forward is not None:
            target = np.zeros(n) if y is None else np.asarray(y)  # preserve dtype
            result = strategy.run_walk_forward(data, target, **walk_forward, X=X)

        else:
            result = strategy.run(data, y, X=X)
    finally:
        if costs is not None and original_cost is not _sentinel:
            strategy.cost = original_cost

    metrics = result.summary(period=period)

    data_block: dict[str, Any] = {
        "kind": getattr(data, "name", None) or type(data).__name__,
        "n": int(n),
        "n_assets": int(prices.shape[1]) if prices.ndim == 2 else 1,
        "start": _index_bound(data, 0),
        "end": _index_bound(data, -1),
        "desc": data_desc,
    }

    if X is None:
        features_block: dict[str, Any] | None = None
    else:
        features_block = {
            "X_shape": [int(d) for d in np.asarray(X).shape],
            "names": list(feature_names) if feature_names is not None else None,
            "desc": feature_desc,
        }

    spec: dict[str, Any] = {
        "data": data_block,
        "features": features_block,
        "model": type(strategy.model).__name__ if getattr(
            strategy, "model", None) is not None else None,
        "signal": _callable_name(getattr(strategy, "signal", None)),
        "walk_forward": walk_forward,
        "cost": type(strategy.cost).__name__ if strategy.cost is not None else None,
        "period": period,
        "seed": seed,
    }

    series: dict[str, list[Any]] = {
        "equity": np.asarray(result.equity, dtype=float).tolist(),
        "returns": np.asarray(result.returns, dtype=float).tolist(),
    }
    # Persist the datetime index (tail-aligned to the curve) so the report can
    # plot against real dates rather than bar numbers; omitted when the series
    # carries no temporal index.
    date_index = _series_index(data, len(series["equity"]))
    if date_index is not None:
        series["index"] = date_index
    # Persist per-asset gross contributions of a multi-asset book so the report
    # can attribute the book return by asset; absent for a single-asset run.
    asset_contrib = getattr(result, "asset_gross_returns", None)
    if asset_contrib is not None:
        series["asset_contrib"] = np.asarray(asset_contrib, dtype=float).tolist()

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
