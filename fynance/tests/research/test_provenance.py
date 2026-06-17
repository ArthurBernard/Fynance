#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for the self-describing provenance block of an experiment.

``run_experiment`` records *what produced the result* (data, features, model,
signal, run config) in ``Experiment.spec`` and ``write_report`` surfaces it as a
Provenance table — backward compatible with older specs.
"""

# Third-party
import numpy as np

# Local
from fynance.research import Experiment, gbm, run_experiment, write_report
from fynance.research.report import _provenance_table
from fynance.strategy import Strategy


def _rule_strategy() -> Strategy:
    return Strategy(features=lambda p: np.diff(p, prepend=p[0]), signal=np.sign)


def test_data_provenance_recorded():
    exp = run_experiment(_rule_strategy(), gbm(300, seed=1), name="d",
                         data_desc="synthetic GBM, daily")
    data = exp.spec["data"]

    assert data["kind"] == "synthetic-gbm"
    assert data["n"] == 300
    assert data["start"] == 0 and data["end"] == 299  # default int index bounds
    assert data["desc"] == "synthetic GBM, daily"


def test_features_provenance_recorded():
    n = 300
    X = np.random.default_rng(0).standard_normal((n, 2)).astype(np.float32)
    exp = run_experiment(
        Strategy(signal=lambda z: np.sign(z[:, 0])), gbm(n, seed=1), name="f",
        X=X, feature_names=["roc", "vol"], feature_desc="2 causal cols",
    )
    feats = exp.spec["features"]

    assert feats["X_shape"] == [n, 2]
    assert feats["names"] == ["roc", "vol"]
    assert feats["desc"] == "2 causal cols"


def test_features_none_without_X():
    exp = run_experiment(_rule_strategy(), gbm(200, seed=2), name="g")

    assert exp.spec["features"] is None
    # The price-only path still records data + run config.
    assert exp.spec["data"]["n"] == 200
    assert exp.spec["seed"] == 0


def test_signal_name_recorded():
    exp = run_experiment(_rule_strategy(), gbm(150, seed=3), name="s")

    assert exp.spec["signal"] == "sign"  # np.sign exposes __name__


def test_provenance_survives_json_roundtrip():
    exp = run_experiment(_rule_strategy(), gbm(150, seed=3), name="r",
                         data_desc="x")
    again = Experiment.from_dict(exp.to_dict())

    assert again.spec == exp.spec


def test_report_renders_provenance_table(tmp_path):
    n = 200
    X = np.random.default_rng(0).standard_normal((n, 1)).astype(np.float32)
    exp = run_experiment(
        Strategy(signal=lambda z: np.sign(z[:, 0])), gbm(n, seed=4), name="rep",
        X=X, feature_names=["mom"], feature_desc="one momentum column",
    )
    write_report(exp, tmp_path, notebook=False)
    text = (tmp_path / "rep" / "report.md").read_text()

    assert "## Provenance" in text
    assert "synthetic-gbm" in text
    assert "X=[200, 1]" in text
    assert "mom" in text
    assert "one momentum column" in text


def test_provenance_table_degrades_on_old_spec():
    # An older experiment.json with a flat data string and no feature keys.
    legacy = {"data": "some-old-series", "walk_forward": None, "seed": 7}
    table = _provenance_table(legacy)

    assert "some-old-series" in table
    assert "none (price-only)" in table  # features absent -> graceful default


def test_provenance_table_empty_spec():
    assert "_no provenance_" in _provenance_table(None)
    assert "_no provenance_" in _provenance_table({})
