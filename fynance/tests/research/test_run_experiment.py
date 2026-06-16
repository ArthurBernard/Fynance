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
