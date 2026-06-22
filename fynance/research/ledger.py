#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Persistent experiment ledger.

A :class:`Ledger` turns one-off runs into **cumulative** research: it persists
experiments under a caller-provided ``root``, reloads them, ranks them into a
leaderboard, and tracks the **number of trials** — which it can feed into the
deflated Sharpe ratio so a selected strategy is judged against the multiple
testing it came from. The store lives entirely under ``root`` (the caller's
private repo) — never inside fynance.

"""

# Built-in
from __future__ import annotations

from pathlib import Path

# Third-party
import numpy as np

# Local
from fynance.research.compare import leaderboard
from fynance.research.experiment import Experiment
from fynance.research.guards import deflated_sharpe_ratio

__all__ = ['Ledger']


class Ledger:
    """ A persistent, append-only store of experiments under ``root``.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory the experiments live under (created on demand). Each experiment
        is stored at ``<root>/<name>/experiment.json``.

    Examples
    --------
    >>> import tempfile
    >>> from fynance.research import Experiment, Ledger
    >>> d = tempfile.mkdtemp()
    >>> led = Ledger(d)
    >>> _ = led.append(Experiment(name="a", metrics={"sharpe": 1.0}))
    >>> _ = led.append(Experiment(name="b", metrics={"sharpe": 2.0}))
    >>> led.n_trials
    2
    >>> [r["name"] for r in led.leaderboard()]
    ['b', 'a']

    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def append(self, experiment: Experiment) -> Path:
        """ Persist ``experiment`` under the ledger root; return its json path.

        The store is **append-only**: appending an experiment whose ``name``
        already exists raises :class:`FileExistsError` rather than silently
        overwriting the prior run. Overwriting would undercount
        :attr:`n_trials` and so deflate the multiple-testing correction fed to
        the deflated Sharpe ratio. Pick a unique ``name`` (e.g. version-suffix
        a re-run) before appending.

        Parameters
        ----------
        experiment : Experiment
            The experiment to persist.

        Returns
        -------
        pathlib.Path
            Path to the written ``experiment.json``.

        Raises
        ------
        FileExistsError
            If an experiment with the same ``name`` is already stored.

        """
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / experiment.name / "experiment.json").exists():
            raise FileExistsError(
                f"experiment {experiment.name!r} already in the ledger; the "
                f"store is append-only — use a unique name for a re-run"
            )

        return experiment.save(self.root)

    def load(self) -> list[Experiment]:
        """ Load every stored experiment (sorted by name for determinism). """
        if not self.root.exists():
            return []

        return [Experiment.load(p)
                for p in sorted(self.root.glob("*/experiment.json"))]

    @property
    def n_trials(self) -> int:
        """ Number of experiments in the ledger (the multiple-testing count). """
        if not self.root.exists():
            return 0

        return sum(1 for _ in self.root.glob("*/experiment.json"))

    def leaderboard(self, *, sort_by: str = "sharpe",
                    descending: bool = True) -> list[dict[str, float | str]]:
        """ Rank the stored experiments (see :func:`fynance.research.leaderboard`). """
        return leaderboard(self.load(), sort_by=sort_by, descending=descending)

    def deflated_sharpe(self, experiment: Experiment,
                        metric: str = "sharpe") -> float:
        """ Deflated Sharpe of ``experiment`` against the ledger's trial count.

        Uses the ledger's :attr:`n_trials` as the number of trials and the
        dispersion of the stored Sharpe metrics as the trial variance, so a
        selected strategy is judged against the multiple testing it came from.

        The stored ``sharpe`` metric is **annualized** (by the ``period`` used
        at run time, recorded in ``spec``), whereas
        :func:`~fynance.research.deflated_sharpe_ratio` expects a
        **per-observation** Sharpe (it scales by ``sqrt(n_obs - 1)``
        internally). This method therefore de-annualizes both the selected
        strategy's Sharpe and the across-trial variance before the call:
        ``sr_obs = sr_annual / sqrt(period)`` and ``var_obs = var_annual /
        period``. Skipping this de-annualization saturates the DSR to ~1
        (a modest per-period edge looks certain), defeating the guard.

        Parameters
        ----------
        experiment : Experiment
            The selected experiment to deflate.
        metric : str
            Metric key holding the (annualized) Sharpe (default ``"sharpe"``).

        Returns
        -------
        float
            Deflated Sharpe ratio in ``[0, 1]``.

        """
        period = self._period(experiment)
        sharpes = [float(e.metrics[metric]) / np.sqrt(self._period(e))
                   for e in self.load() if metric in e.metrics]
        sr_variance = float(np.var(sharpes)) if len(sharpes) > 1 else 1.0

        n_obs = len(experiment.series["returns"]) if (
            experiment.series and experiment.series.get("returns")) else 0

        return deflated_sharpe_ratio(
            float(experiment.metrics[metric]) / np.sqrt(period), n_obs,
            max(self.n_trials, 1), sr_variance=sr_variance,
        )

    @staticmethod
    def _period(experiment: Experiment) -> float:
        """ Annualization factor used to compute ``experiment``'s Sharpe. """
        period = experiment.spec.get("period") if experiment.spec else None
        try:
            value = float(period)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 252.0

        return value if value > 0.0 else 252.0
