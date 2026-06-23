#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for :func:`fynance.research.write_report`. """

# Built-in
import importlib.util
from pathlib import Path
from types import SimpleNamespace

# Third-party
import numpy as np
import pytest

# Local
import fynance
from fynance.research import gbm, run_experiment, write_report
from fynance.strategy import Strategy

_HAS_NBFORMAT = importlib.util.find_spec("nbformat") is not None
_HAS_KERNEL = importlib.util.find_spec("ipykernel") is not None


@pytest.fixture
def experiment(tmp_path):
    """ A real experiment from a synthetic run. """
    strat = Strategy(features=lambda p: np.diff(p, prepend=p[0]))
    return run_experiment(strat, gbm(400, seed=7), name="demo")


def test_writes_markdown_and_png(tmp_path, experiment):
    out = write_report(experiment, tmp_path, notebook=False)

    md = tmp_path / "demo" / "report.md"
    png = tmp_path / "demo" / "tearsheet.png"
    assert out["markdown"] == md and md.is_file()
    assert out["png"] == png and png.is_file() and png.stat().st_size > 0

    text = md.read_text()
    assert "# Experiment: demo" in text
    assert "tearsheet.png" in text
    # at least one metric value rendered
    assert "sharpe" in text


def test_nothing_written_outside_output_dir(tmp_path, experiment):
    pkg = Path(fynance.__file__).resolve().parent
    before = {p for p in pkg.rglob("report.md")}

    write_report(experiment, tmp_path)

    assert {p for p in pkg.rglob("report.md")} == before


def test_import_fynance_stays_matplotlib_free():
    import subprocess
    import sys

    code = "import fynance, sys; print('matplotlib' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "False"


@pytest.mark.skipif(not _HAS_NBFORMAT, reason="nbformat not installed")
def test_notebook_written(tmp_path, experiment):
    out = write_report(experiment, tmp_path, notebook=True)
    nb = tmp_path / "demo" / "report.ipynb"

    assert out["notebook"] == nb and nb.is_file()

    import nbformat

    parsed = nbformat.read(nb, as_version=4)
    assert any("tearsheet" in c.source for c in parsed.cells)


@pytest.mark.skipif(not (_HAS_NBFORMAT and _HAS_KERNEL),
                    reason="needs nbformat + a jupyter kernel")
def test_notebook_execution(tmp_path, experiment):
    # Execute against the saved experiment.json (the notebook reads it relatively).
    experiment.save(tmp_path, name="demo")
    out = write_report(experiment, tmp_path, notebook=True, execute=True)

    import nbformat

    parsed = nbformat.read(out["notebook"], as_version=4)
    code_cells = [c for c in parsed.cells if c.cell_type == "code"]
    assert any(c.get("outputs") for c in code_cells)


@pytest.fixture
def dated_experiment():
    """ A real experiment whose price series carries a datetime index. """
    from fynance.core import PriceSeries

    n = 400
    values = gbm(n, seed=7).to_numpy()
    dates = np.datetime64("2019-01-01") + np.arange(n)
    ps = PriceSeries(values, index=dates, name="dated")
    strat = Strategy(features=lambda p: np.diff(p, prepend=p[0]))

    return run_experiment(strat, ps, name="dated")


def test_report_uses_date_axis_when_indexed(tmp_path, dated_experiment):
    # The dated experiment persists an index; the tearsheet PNG is written from
    # it without error (drawn against dates rather than bar numbers).
    assert "index" in dated_experiment.series

    out = write_report(dated_experiment, tmp_path, notebook=False)
    png = tmp_path / "dated" / "tearsheet.png"

    assert out["png"] == png and png.is_file() and png.stat().st_size > 0


def test_tearsheet_plots_dates_for_indexed_curve():
    # Threading a datetime index through to the plot layer yields a date axis
    # (datetime64 x-data + a date locator) instead of integer bar numbers.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates

    from fynance.plot.equity import plot_equity

    dates = np.datetime64("2021-01-01") + np.arange(60)
    equity = np.linspace(1.0, 1.3, dates.size)

    ax = plot_equity(SimpleNamespace(equity=equity, index=dates))
    xdata = np.asarray(ax.get_lines()[0].get_xdata())

    assert np.issubdtype(xdata.dtype, np.datetime64)
    assert isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
