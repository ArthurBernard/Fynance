#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :class:`fynance.research.Ledger`. """

# Built-in
from pathlib import Path

# Third-party
import numpy as np
import pytest

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


def test_append_rejects_duplicate_name(tmp_path):
    # Append-only: a name collision must raise rather than silently overwrite
    # (which would undercount n_trials and deflate the DSR correction).
    led = Ledger(tmp_path)
    led.append(Experiment(name="dup", metrics={"sharpe": 1.0}))

    with pytest.raises(FileExistsError):
        led.append(Experiment(name="dup", metrics={"sharpe": 9.0}))

    # The original is untouched and the trial count stays correct.
    assert led.n_trials == 1
    assert led.load()[0].metrics["sharpe"] == 1.0


def test_deflated_sharpe_deannualizes_not_saturated(tmp_path):
    # Realistic annualized Sharpes (~1) over n_obs ~ 500 must NOT saturate the
    # DSR to ~1 once de-annualized; before the fix the annualized Sharpe was
    # fed straight in, pinning the DSR at ~1.
    led = Ledger(tmp_path)
    rng = np.random.default_rng(0)
    for i in range(12):
        n = 500
        # Per-day Sharpe ~ 0.06 -> annualized ~ 1.0, with spread across trials.
        per_day = 0.06 + 0.02 * rng.standard_normal()
        sharpe_annual = per_day * np.sqrt(252)
        returns = (per_day + rng.standard_normal(n)).tolist()
        led.append(Experiment(
            name=f"t{i}",
            spec={"period": 252},
            metrics={"sharpe": float(sharpe_annual)},
            series={"returns": returns},
        ))

    best = max(led.load(), key=lambda e: e.metrics["sharpe"])
    dsr = led.deflated_sharpe(best)

    assert 0.0 <= dsr <= 1.0
    assert dsr < 0.999  # NOT saturated — the multiple-testing correction bites


def test_deflated_sharpe_uses_recorded_period(tmp_path):
    # The de-annualization must use the period recorded in spec. A monthly run
    # (period=12) and a daily run (period=252) carrying the SAME annualized
    # Sharpe de-annualize to different per-obs Sharpes -> different DSR.
    returns = np.random.default_rng(1).standard_normal(400).tolist()

    def _led(period):
        led = Ledger(tmp_path / f"p{period}")
        e = Experiment(name="x", spec={"period": period},
                       metrics={"sharpe": 1.5}, series={"returns": returns})
        led.append(e)
        return led, e

    led_d, e_d = _led(252)
    led_m, e_m = _led(12)

    assert led_d.deflated_sharpe(e_d) != led_m.deflated_sharpe(e_m)
