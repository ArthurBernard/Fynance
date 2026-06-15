#!/usr/bin/env python3
# coding: utf-8

""" Smoke tests for the Streamlit playground helpers (no Streamlit runtime). """

# Built-in packages
import sys
from pathlib import Path

# Third-party packages
import numpy as np

# Make the repo-root ``apps`` package importable.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Local packages
from apps.playground.runner import (  # noqa: E402
    TEMPLATE,
    compile_signal,
    run_signal,
)
from fynance.backtest import BacktestResult  # noqa: E402


def _prices(n=200):
    rng = np.random.default_rng(0)
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))


def test_template_compiles_and_runs():
    signal = compile_signal(TEMPLATE)
    res = run_signal(_prices(), signal)
    assert isinstance(res, BacktestResult)
    assert np.isfinite(res.summary()["sharpe"])


def test_compile_rejects_missing_signal():
    import pytest
    with pytest.raises(ValueError):
        compile_signal("x = 1")


def test_run_signal_with_fee_charges_cost():
    code = "import numpy as np\n\ndef signal(prices):\n    return np.resize([1.0, -1.0], len(prices))\n"
    signal = compile_signal(code)
    res = run_signal(_prices(), signal, fee=0.01)
    assert res.summary()["total_cost"] > 0.0
