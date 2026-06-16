#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :class:`fynance.research.Ledger`. """

# Built-in
from pathlib import Path

# Third-party
import numpy as np

# Local
import fynance
from fynance.research import Experiment, Ledger, gbm, run_experiment
from fynance.strategy import Strategy


def _exp(name, seed):
    strat = Strategy(features=lambda p: np.diff(p, prepend=p[0]))
    return run_experiment(strat, gbm(400, seed=seed), name=name, seed=seed)


def test_append_load_roundtrip(tmp_path):
    led = Ledger(tmp_path)
    led.append(_exp("a", 1))
    led.append(_exp("b", 2))

    loaded = led.load()
    assert {e.name for e in loaded} == {"a", "b"}
    assert led.n_trials == 2
    assert (tmp_path / "a" / "experiment.json").is_file()


def test_empty_ledger(tmp_path):
    led = Ledger(tmp_path / "nope")
    assert led.load() == []
    assert led.n_trials == 0


def test_leaderboard_ranked(tmp_path):
    led = Ledger(tmp_path)
    for nm, sd in [("a", 1), ("b", 2), ("c", 3)]:
        led.append(_exp(nm, sd))

    rows = led.leaderboard(sort_by="sharpe")
    sharpes = [r["sharpe"] for r in rows]
    assert sharpes == sorted(sharpes, reverse=True)


def test_deflated_sharpe_uses_trial_count(tmp_path):
    led = Ledger(tmp_path)
    exps = [_exp(n, s) for n, s in [("a", 1), ("b", 2), ("c", 3), ("d", 4)]]
    for e in exps:
        led.append(e)

    best = max(exps, key=lambda e: e.metrics["sharpe"])
    dsr = led.deflated_sharpe(best)
    assert 0.0 <= dsr <= 1.0


def test_store_stays_under_root(tmp_path):
    pkg = Path(fynance.__file__).resolve().parent
    before = {p for p in pkg.rglob("experiment.json")}

    Ledger(tmp_path).append(_exp("x", 1))

    assert {p for p in pkg.rglob("experiment.json")} == before


def test_n_trials_counts_only_experiments(tmp_path):
    # A stray sub-dir without experiment.json must not inflate the count.
    led = Ledger(tmp_path)
    led.append(Experiment(name="real", metrics={"sharpe": 1.0}))
    (tmp_path / "stray").mkdir()
    (tmp_path / "stray" / "report.md").write_text("noise")

    assert led.n_trials == 1
