#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Portable report writer.

Renders an :class:`~fynance.research.Experiment` into remotely-viewable
artifacts under a caller-provided ``output_dir``: a markdown summary with an
embedded tearsheet PNG (viewable on GitHub from a phone) and a re-runnable
notebook. matplotlib and nbformat are imported lazily so ``import fynance`` stays
matplotlib-free and the package does not hard-require Jupyter.

"""

# Built-in
from __future__ import annotations

import warnings
from pathlib import Path

# Third-party
import numpy as np

# Local
from fynance.research.experiment import Experiment

__all__ = ['write_report']


def _metrics_table(metrics: dict[str, float]) -> str:
    """ Render a metrics dict as a markdown table. """
    if not metrics:
        return "_no metrics_\n"

    lines = ["| metric | value |", "| --- | --- |"]
    lines += [f"| {k} | {v:.4f} |" for k, v in metrics.items()]

    return "\n".join(lines) + "\n"


def _provenance_table(spec: dict | None) -> str:
    """ Render the experiment ``spec`` provenance as a markdown table.

    Surfaces *what produced the result*: data, features, model, signal, and the
    run config. Degrades gracefully to whatever fields are present (older specs).
    """
    if not spec:
        return "_no provenance_\n"

    rows: list[tuple[str, str]] = []

    data = spec.get("data")
    if isinstance(data, dict):
        span = ""
        if data.get("start") is not None or data.get("end") is not None:
            span = f" ({data.get('start')} → {data.get('end')})"
        rows.append(("data", f"{data.get('kind')} · n={data.get('n')}{span}"))
        if data.get("desc"):
            rows.append(("data desc", str(data["desc"])))
    elif data is not None:
        rows.append(("data", str(data)))

    feats = spec.get("features")
    if isinstance(feats, dict):
        rows.append(("features", f"X={feats.get('X_shape')}"))
        if feats.get("names"):
            rows.append(("feature names", ", ".join(map(str, feats["names"]))))
        if feats.get("desc"):
            rows.append(("feature desc", str(feats["desc"])))
    else:
        rows.append(("features", "none (price-only)"))

    for key in ("model", "signal", "walk_forward", "cost", "period", "seed"):
        if key in spec and spec[key] is not None:
            rows.append((key, f"`{spec[key]}`"))

    lines = ["| field | value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rows]

    return "\n".join(lines) + "\n"


def _write_png(experiment: Experiment, png_path: Path, period: int) -> bool:
    """ Render the tearsheet from the equity curve to ``png_path``.

    Returns True if written, False if there is no equity curve to plot.
    """
    if not experiment.series or not experiment.series.get("equity"):
        return False

    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    from fynance.plot import tearsheet

    equity = np.asarray(experiment.series["equity"], dtype=float)
    fig = tearsheet(equity, period=period)
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return True


def _build_notebook(experiment: Experiment, target: Path, period: int,
                    execute: bool) -> Path | None:
    """ Write a re-runnable notebook reconstructing the report. Returns its path
    or None when nbformat is unavailable. """
    try:
        import nbformat
        from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
    except ImportError:
        warnings.warn("nbformat not installed — skipping notebook generation.",
                      stacklevel=2)
        return None

    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(f"# Experiment: {experiment.name}\n\n"
                          f"Reconstructed from `experiment.json` "
                          f"(fynance {experiment.fynance_version}). The full "
                          f"provenance (data, features, model, run config) lives "
                          f"under `exp['spec']`."),
        new_code_cell(
            "import json\n"
            "import numpy as np\n"
            "from fynance.plot import tearsheet\n"
            "\n"
            "exp = json.load(open('experiment.json'))\n"
            "equity = np.asarray(exp['series']['equity'], dtype=float)\n"
            f"fig = tearsheet(equity, period={period})\n"
            "fig"
        ),
    ]

    nb_path = target / "report.ipynb"

    if execute:
        try:
            from nbclient import NotebookClient

            NotebookClient(nb, resources={"metadata": {"path": str(target)}}).execute()
        except Exception as exc:  # noqa: BLE001 — degrade gracefully (no kernel, etc.)
            warnings.warn(f"notebook execution skipped: {exc!r}", stacklevel=2)

    nbformat.write(nb, nb_path)

    return nb_path


def write_report(
    experiment: Experiment,
    output_dir: str | Path,
    *,
    notebook: bool = True,
    execute: bool = False,
) -> dict[str, Path]:
    """ Write a portable report for ``experiment`` under ``output_dir``.

    Creates ``<output_dir>/<experiment.name>/`` with ``report.md`` and a
    ``tearsheet.png`` (when an equity curve is present), plus ``report.ipynb``
    when ``notebook`` is True. Nothing is ever written outside ``output_dir``.

    Parameters
    ----------
    experiment : Experiment
        The experiment to render.
    output_dir : str or pathlib.Path
        Base directory for the artifacts.
    notebook : bool
        Also emit a re-runnable ``report.ipynb`` (needs ``nbformat``).
    execute : bool
        Execute the notebook before writing it (needs ``nbclient`` + a kernel);
        degrades to an unexecuted notebook with a warning if unavailable.

    Returns
    -------
    dict of str to pathlib.Path
        The written artifacts (keys: ``markdown``, ``png``, ``notebook`` —
        present only when actually written).

    """
    period = int(experiment.spec.get("period", 252)) if experiment.spec else 252
    target = Path(output_dir) / experiment.name
    target.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    png_path = target / "tearsheet.png"
    if _write_png(experiment, png_path, period):
        written["png"] = png_path

    md = [f"# Experiment: {experiment.name}\n",
          f"- fynance: `{experiment.fynance_version}`",
          f"- created: `{experiment.created_at}`",
          f"- seed: `{experiment.seed}`",
          "\n## Provenance\n",
          _provenance_table(experiment.spec),
          "\n## Metrics\n",
          _metrics_table(experiment.metrics)]
    if "png" in written:
        md += ["\n## Tearsheet\n", "![tearsheet](tearsheet.png)\n"]

    md_path = target / "report.md"
    md_path.write_text("\n".join(line for line in md if line != "") + "\n",
                       encoding="utf-8")
    written["markdown"] = md_path

    if notebook:
        nb_path = _build_notebook(experiment, target, period, execute)
        if nb_path is not None:
            written["notebook"] = nb_path

    return written
