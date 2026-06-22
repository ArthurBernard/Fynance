#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Multi-strategy comparison report.

Ranks a set of :class:`~fynance.research.Experiment` runs into a leaderboard and
overlays their equity curves — the artifact for *"which of these strategies is
best, and by how much vs a baseline?"*. Written, like everything in the harness,
only to a caller-provided ``output_dir``.

"""

# Built-in
from __future__ import annotations

from pathlib import Path

# Third-party
import numpy as np

# Local
from fynance.research.experiment import Experiment

__all__ = ['compare_report', 'leaderboard']


def leaderboard(
    experiments: list[Experiment],
    *,
    sort_by: str = "sharpe",
    descending: bool = True,
) -> list[dict[str, float | str]]:
    """ Rank experiments into leaderboard rows.

    Parameters
    ----------
    experiments : list of Experiment
        The runs to rank.
    sort_by : str
        Metric key to sort on. Rows missing the metric **or** holding a NaN for
        it sink to the bottom of the ranking.
    descending : bool
        Higher is better when True.

    Returns
    -------
    list of dict
        One ``{"name": ..., <metric>: <value>, ...}`` row per experiment, ranked.

    """
    rows: list[dict[str, float | str]] = [
        {"name": e.name, **{k: float(v) for k, v in e.metrics.items()}}
        for e in experiments
    ]
    # NaN must rank as "worst", not float to the top: a present-but-NaN metric
    # is treated like a missing value (the docstring's promise).
    worst = float("-inf") if descending else float("inf")

    def _key(row: dict[str, float | str]) -> float:
        value = row.get(sort_by, worst)
        if isinstance(value, float) and np.isnan(value):
            return worst

        return value  # type: ignore[return-value]

    rows.sort(key=_key, reverse=descending)

    return rows


def _write_overlay(experiments: list[Experiment], png_path: Path) -> bool:
    """ Overlay each experiment's equity curve (rebased to 100). """
    curves = [(e.name, e.series["equity"]) for e in experiments
              if e.series and e.series.get("equity")]
    if not curves:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, equity in curves:
        eq = np.asarray(equity, dtype=float)
        ax.plot(100.0 * eq / eq[0], label=label)
    ax.set_title("Equity comparison (rebased to 100)")
    ax.set_xlabel("step")
    ax.set_ylabel("equity")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return True


def _markdown_table(rows: list[dict[str, float | str]]) -> str:
    """ Render leaderboard rows as a markdown table. """
    if not rows:
        return "_no experiments_\n"

    cols = ["name"] + [k for k in rows[0] if k != "name"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for r in rows:
        cells = [str(r["name"])] + [
            f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
            for c in cols[1:]
        ]
        body.append("| " + " | ".join(cells) + " |")

    return "\n".join([head, sep, *body]) + "\n"


def compare_report(
    experiments: list[Experiment],
    output_dir: str | Path,
    *,
    name: str = "comparison",
    sort_by: str = "sharpe",
) -> dict[str, Path]:
    """ Write a leaderboard + equity-overlay comparison report.

    Creates ``<output_dir>/<name>/`` with ``report.md`` (a leaderboard table
    ranked by ``sort_by``) and ``equity_overlay.png`` (when curves are present).
    The first experiment is treated as the baseline in the narrative.

    Parameters
    ----------
    experiments : list of Experiment
        The runs to compare (≥ 1).
    output_dir : str or pathlib.Path
        Base directory for the artifacts.
    name : str
        Sub-directory name.
    sort_by : str
        Leaderboard sort metric.

    Returns
    -------
    dict of str to pathlib.Path
        Written artifacts (keys: ``markdown``, ``png`` when produced).

    """
    if not experiments:
        raise ValueError("compare_report needs at least one experiment")

    target = Path(output_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    png = target / "equity_overlay.png"
    if _write_overlay(experiments, png):
        written["png"] = png

    rows = leaderboard(experiments, sort_by=sort_by)
    md = [f"# Comparison: {name}\n",
          f"Ranked by `{sort_by}` — {len(experiments)} experiment(s), "
          f"baseline `{experiments[0].name}`.\n",
          "## Leaderboard\n",
          _markdown_table(rows)]
    if "png" in written:
        md += ["\n## Equity overlay\n", "![equity overlay](equity_overlay.png)\n"]

    md_path = target / "report.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    written["markdown"] = md_path

    return written
