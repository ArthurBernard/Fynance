#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :func:`fynance.research.run_experiment`. """

# Built-in
from pathlib import Path

# Third-party
import numpy as np

# Local
import fynance
from fynance.backtest import ProportionalCost
from fynance.core import PriceSeries
from fynance.research import Experiment, gbm, run_experiment
from fynance.strategy import Strategy


def momentum() -> Strategy:
    """ A simple causal rule-based momentum strategy. """
    return Strategy(
        features=lambda p: np.diff(p, prepend=p[0]),
        cost=ProportionalCost(0.0005),
    )


def test_returns_populated_experiment(tmp_path):
    exp = run_experiment(momentum(), gbm(500, seed=7), name="m1",
                         output_dir=tmp_path)

    assert isinstance(exp, Experiment)
    assert exp.metrics and all(np.isfinite(list(exp.metrics.values())))
    assert exp.series["equity"] and exp.series["returns"]
    assert (tmp_path / "m1" / "experiment.json").is_file()
    # Round-trips through disk.
    assert Experiment.load(tmp_path / "m1" / "experiment.json").metrics == exp.metrics


def test_reproducible_same_seed():
    a = run_experiment(momentum(), gbm(500, seed=7), name="a")
    b = run_experiment(momentum(), gbm(500, seed=7), name="b")

    assert a.metrics == b.metrics


def test_output_dir_none_writes_nothing():
    pkg = Path(fynance.__file__).resolve().parent
    before = set(pkg.rglob("experiment.json"))

    exp = run_experiment(momentum(), gbm(200, seed=1), name="x")

    assert isinstance(exp, Experiment)
    assert set(pkg.rglob("experiment.json")) == before


def test_priceseries_and_ndarray_equivalent():
    ps = gbm(300, seed=5)
    a = run_experiment(momentum(), ps, name="ps")
    b = run_experiment(momentum(), ps.to_numpy(), name="np")

    assert a.metrics == b.metrics


def test_costs_override_changes_result():
    data = gbm(400, seed=3)
    free = run_experiment(Strategy(features=lambda p: np.diff(p, prepend=p[0])),
                          data, name="free", costs=ProportionalCost(0.0))
    pricey = run_experiment(Strategy(features=lambda p: np.diff(p, prepend=p[0])),
                            data, name="pricey", costs=ProportionalCost(0.01))

    assert free.metrics != pricey.metrics


def test_costs_override_does_not_mutate_strategy():
    # The override must be confined to the single run: a later run on the same
    # strategy must not inherit it (an aliasing hazard in permutation loops).
    own = ProportionalCost(0.0005)
    strat = Strategy(features=lambda p: np.diff(p, prepend=p[0]), cost=own)
    data = gbm(400, seed=3)

    overridden = run_experiment(strat, data, name="ovr", costs=ProportionalCost(0.02))

    assert strat.cost is own  # restored, not left pointing at the override

    # And a subsequent run with no override uses the strategy's own cost.
    again = run_experiment(strat, data, name="again")
    baseline = run_experiment(
        Strategy(features=lambda p: np.diff(p, prepend=p[0]), cost=ProportionalCost(0.0005)),
        data, name="base",
    )
    assert again.metrics == baseline.metrics
    assert again.metrics != overridden.metrics


def test_no_lookahead_walk_forward():
    # Perturbing the tail must not change the out-of-sample returns on the
    # earlier (unperturbed) prefix — a black-box causality probe.
    data = gbm(600, seed=7).to_numpy()
    wf = {"train": 100, "test": 50}

    base = run_experiment(momentum(), data, name="base", walk_forward=wf)

    pert = data.copy()
    cut = int(len(data) * 0.7)
    rng = np.random.default_rng(0)
    bumps = 1.0 + rng.standard_normal(len(data) - cut) * 0.05
    pert[cut:] = pert[cut - 1] * np.cumprod(bumps)

    pert_exp = run_experiment(momentum(), pert, name="pert", walk_forward=wf)

    k = len(base.series["returns"]) // 3
    assert np.allclose(base.series["returns"][:k], pert_exp.series["returns"][:k])


def test_datetime_index_captured_tail_aligned():
    # A PriceSeries carrying a real datetime index makes run_experiment persist a
    # tail-aligned ISO date index, so the report plots against dates not bars.
    n = 300
    values = gbm(n, seed=11).to_numpy()
    dates = np.datetime64("2020-01-01") + np.arange(n)
    ps = PriceSeries(values, index=dates, name="dated")

    exp = run_experiment(momentum(), ps, name="dated")

    idx = exp.series.get("index")
    assert idx is not None
    # One date per equity point, right-aligned (ends on the last observation).
    assert len(idx) == len(exp.series["equity"])
    assert np.datetime64(idx[-1]) == dates[-1]
    assert np.datetime64(idx[0]) == dates[-len(idx)]
    # ISO strings survive the JSON round-trip.
    assert Experiment.from_dict(exp.to_dict()).series["index"] == idx


def test_non_datetime_index_yields_no_date_axis():
    # gbm's default 0..n-1 integer index is not temporal: no index is stored, so
    # the report falls back to bar numbers (unchanged behavior).
    exp = run_experiment(momentum(), gbm(200, seed=2), name="bars")

    assert "index" not in exp.series
