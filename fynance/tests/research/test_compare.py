#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :mod:`fynance.research.compare`. """

# Built-in
import math
from pathlib import Path

# Third-party
import numpy as np
import pytest

# Local
import fynance
from fynance.research import (
    Experiment,
    compare_report,
    gbm,
    leaderboard,
    run_experiment,
)
from fynance.strategy import Strategy


def _exp(name, seed, fee=0.0):
    from fynance.backtest import ProportionalCost

    strat = Strategy(features=lambda p: np.diff(p, prepend=p[0]),
                     cost=ProportionalCost(fee))
    return run_experiment(strat, gbm(400, seed=seed), name=name, seed=seed)


@pytest.fixture
def experiments():
    return [_exp("a", 1), _exp("b", 2), _exp("c", 3)]


def test_leaderboard_ranked_by_sharpe(experiments):
    rows = leaderboard(experiments, sort_by="sharpe")

    assert {r["name"] for r in rows} == {"a", "b", "c"}
    sharpes = [r["sharpe"] for r in rows]
    assert sharpes == sorted(sharpes, reverse=True)


def test_leaderboard_nan_sinks_to_bottom():
    # A present-but-NaN metric must sort as "worst", not float to the top.
    exps = [
        Experiment(name="good", metrics={"sharpe": 2.0}),
        Experiment(name="bad", metrics={"sharpe": float("nan")}),
        Experiment(name="mid", metrics={"sharpe": 1.0}),
    ]

    rows = leaderboard(exps, sort_by="sharpe", descending=True)

    assert [r["name"] for r in rows[:2]] == ["good", "mid"]
    assert rows[-1]["name"] == "bad"  # NaN sank to the bottom
    assert math.isnan(rows[-1]["sharpe"])


def test_leaderboard_nan_sinks_ascending():
    # Ascending (lower is better): NaN must still sink to the bottom.
    exps = [
        Experiment(name="bad", metrics={"sharpe": float("nan")}),
        Experiment(name="best", metrics={"sharpe": -1.0}),
        Experiment(name="worst", metrics={"sharpe": 5.0}),
    ]

    rows = leaderboard(exps, sort_by="sharpe", descending=False)

    assert rows[0]["name"] == "best"
    assert rows[-1]["name"] == "bad"


def test_compare_report_writes_md_and_png(tmp_path, experiments):
    out = compare_report(experiments, tmp_path, name="cmp")

    md = tmp_path / "cmp" / "report.md"
    png = tmp_path / "cmp" / "equity_overlay.png"
    assert out["markdown"] == md and md.is_file()
    assert out["png"] == png and png.is_file() and png.stat().st_size > 0

    text = md.read_text()
    assert "# Comparison: cmp" in text
    for nm in ("a", "b", "c"):
        assert f"| {nm} |" in text
    assert "equity_overlay.png" in text


def test_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        compare_report([], tmp_path)


def test_nothing_written_outside_output_dir(tmp_path, experiments):
    pkg = Path(fynance.__file__).resolve().parent
    before = {p for p in pkg.rglob("equity_overlay.png")}

    compare_report(experiments, tmp_path)

    assert {p for p in pkg.rglob("equity_overlay.png")} == before
