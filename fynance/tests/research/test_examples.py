#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Smoke test for the shipped ``examples/research_workflow.py``.

Runs the canonical workflow end-to-end on synthetic data into a tmp dir and
asserts the portable artifacts exist — so the documented example cannot silently
rot.
"""

# Built-in
import importlib.util
from pathlib import Path

# Third-party
import numpy as np

# Local
import fynance

_EXAMPLE = (Path(fynance.__file__).resolve().parent.parent
            / "examples" / "research_workflow.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("research_workflow", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_example_file_exists():
    assert _EXAMPLE.is_file()


def test_example_runs_end_to_end(tmp_path):
    module = _load_example()
    out = module.main(output_dir=tmp_path)

    # Both strategies produced finite metrics.
    assert np.isfinite(out["baseline"]["sharpe"])
    assert np.isfinite(out["objective"]["sharpe"])
    assert 0.0 <= out["permutation"]["p_value"] <= 1.0

    # Portable artifacts landed under the caller-provided output_dir.
    for name in ("ts_momentum", "sharpe_nn"):
        assert (tmp_path / name / "report.md").is_file()
        assert (tmp_path / name / "experiment.json").is_file()

    # The report carries the provenance (data + features visible).
    text = (tmp_path / "sharpe_nn" / "report.md").read_text()
    assert "## Provenance" in text
    assert "roc_1" in text  # feature names surfaced
