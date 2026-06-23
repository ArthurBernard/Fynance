#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" The :class:`Experiment` record.

A serializable description of one strategy experiment — enough to reproduce it
and to render a report — with **no run logic** (that lives in
:mod:`fynance.research.runner`). This is the portable artifact a downstream
(private) research repo stores; ``fynance`` itself never persists results except
to a caller-provided ``output_dir``.

"""

# Built-in
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ['Experiment']


def _utc_now() -> str:
    """ Current UTC timestamp as an ISO-8601 string. """
    return datetime.now(timezone.utc).isoformat()


def _fynance_version() -> str:
    """ Resolve the installed fynance version lazily (avoids import cycles). """
    from fynance import __version__

    return __version__


@dataclass
class Experiment:
    """ Serializable record of a single strategy experiment.

    Parameters
    ----------
    name : str
        Short slug identifying the experiment (also the output sub-directory).
    spec : dict, optional
        Everything needed to reproduce the run: strategy description, params,
        walk-forward config, cost config and the data spec (e.g.
        ``{"kind": "synthetic-gbm", "seed": 7, "n": 1000}``).
    code : str, optional
        The generated strategy source, stored verbatim for audit/reproduction.
    seed : int
        Master seed driving every stochastic step.
    metrics : dict of str to float, optional
        Summary metrics (``sharpe``/``sortino``/…); empty until a run fills it.
    series : dict of str to list, optional
        JSON-friendly curves (e.g. ``equity``/``returns`` of float, plus an
        optional ``index`` of ISO-8601 date strings) for the report.
    created_at : str
        ISO-8601 UTC creation time (set automatically).
    fynance_version : str
        Version of fynance that produced the experiment (set automatically).

    Examples
    --------
    >>> from fynance.research import Experiment
    >>> e = Experiment(name="demo", seed=7, metrics={"sharpe": 1.5})
    >>> e2 = Experiment.from_dict(e.to_dict())
    >>> e2.name, e2.seed, e2.metrics["sharpe"]
    ('demo', 7, 1.5)

    """

    name: str
    spec: dict[str, Any] = field(default_factory=dict)
    code: str | None = None
    seed: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    series: dict[str, list[Any]] | None = None
    created_at: str = field(default_factory=_utc_now)
    fynance_version: str = field(default_factory=_fynance_version)

    def to_dict(self) -> dict[str, Any]:
        """ Return a JSON-serializable dict of the experiment. """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        """ Rebuild an :class:`Experiment` from a :meth:`to_dict` mapping.

        Unknown keys are **ignored** rather than raising, so an artifact written
        by a newer fynance (carrying extra fields) still loads on an older one.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict` (extra keys tolerated).

        Returns
        -------
        Experiment
            The rebuilt experiment.

        """
        known = {f.name for f in fields(cls)}

        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, output_dir: str | Path, *, name: str | None = None) -> Path:
        """ Write ``<output_dir>/<name>/experiment.json`` and return its path.

        Always writes under the caller-provided ``output_dir`` — never inside
        the package. Parent directories are created as needed.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Base directory the artifact is written under.
        name : str, optional
            Sub-directory name; defaults to :attr:`name`.

        Returns
        -------
        pathlib.Path
            Path to the written ``experiment.json``.

        """
        sub = name if name is not None else self.name
        target = Path(output_dir) / sub
        target.mkdir(parents=True, exist_ok=True)
        path = target / "experiment.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

        return path

    @classmethod
    def load(cls, path: str | Path) -> Experiment:
        """ Load an :class:`Experiment` from an ``experiment.json`` file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the ``experiment.json`` artifact.

        Returns
        -------
        Experiment
            The loaded experiment.

        Raises
        ------
        ValueError
            If the file is not valid JSON (corrupt or truncated) — re-raised
            with the offending path for a clear diagnostic.

        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"corrupt or truncated experiment artifact at {path}: {exc}"
            ) from exc

        return cls.from_dict(data)
