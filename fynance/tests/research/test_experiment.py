#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :class:`fynance.research.Experiment`. """

# Built-in
import json
from pathlib import Path

# Third-party
import pytest

# Local
import fynance
from fynance.research import Experiment


@pytest.fixture
def populated():
    """ A fully-populated experiment. """
    return Experiment(
        name="demo",
        spec={"kind": "synthetic-gbm", "seed": 7, "n": 5},
        code="def strat(p): return p",
        seed=7,
        metrics={"sharpe": 1.5, "max_drawdown": -0.1},
        series={"equity": [100.0, 101.0, 99.5, 102.0]},
    )


def test_to_from_dict_round_trip(populated):
    restored = Experiment.from_dict(populated.to_dict())

    assert restored.to_dict() == populated.to_dict()
    assert restored.metrics["sharpe"] == 1.5
    assert restored.series["equity"][0] == 100.0


def test_to_dict_is_json_serializable(populated):
    # Must round-trip through real JSON without custom encoders.
    text = json.dumps(populated.to_dict())
    assert Experiment.from_dict(json.loads(text)).name == "demo"


def test_save_and_load(tmp_path, populated):
    path = populated.save(tmp_path)

    assert path == tmp_path / "demo" / "experiment.json"
    assert path.is_file()

    loaded = Experiment.load(path)
    assert loaded.to_dict() == populated.to_dict()


def test_save_custom_name(tmp_path, populated):
    path = populated.save(tmp_path, name="run-42")

    assert path == tmp_path / "run-42" / "experiment.json"
    assert path.is_file()


def test_save_never_writes_inside_package(tmp_path, populated):
    pkg_root = Path(fynance.__file__).resolve().parent
    before = {p for p in pkg_root.rglob("experiment.json")}

    populated.save(tmp_path)

    after = {p for p in pkg_root.rglob("experiment.json")}
    assert before == after  # nothing new written under the package


def test_version_and_timestamp_autoset():
    e = Experiment(name="x")

    assert e.fynance_version == fynance.__version__
    assert e.created_at  # non-empty ISO timestamp
