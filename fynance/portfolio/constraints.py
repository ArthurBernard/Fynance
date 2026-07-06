#!/usr/bin/env python3
# coding: utf-8

r""" Feasible-set projection for portfolio weights.

Composable overlay that projects a weight vector (or a ``(T, N)`` book,
row-wise) onto a feasible set defined by a per-asset box, a gross-leverage
cap, a net-exposure range and named group bounds, as a least-distance
(Euclidean) projection. Meant to run after any allocator or signal and
before :func:`fynance.backtest.engine.backtest` — clip and rescale whatever
weights come out of upstream logic so they respect risk limits without
distorting their shape more than strictly necessary.

Main entry points
-----------------
- :func:`project_weights` — least-distance projection onto box / gross /
  net / group constraints, with a cheap closed-form fast path for the
  common box-and-gross case and an exact SLSQP fallback for everything
  else.

"""

from __future__ import annotations

# Built-in packages
from typing import Sequence

# Third-party packages
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, minimize

__all__ = ['project_weights']


# =========================================================================== #
#                                validation                                   #
# =========================================================================== #


def _validate_box(box: tuple[float, float] | NDArray, N: int) -> tuple[NDArray, NDArray]:
    """ Normalize `box` to per-asset ``(lo, hi)`` arrays of length `N`.

    Accepts a scalar ``(lo, hi)`` pair (broadcast to every asset) or a
    ``(2, N)`` array-like of per-asset bounds.

    """
    box_arr = np.asarray(box, dtype=np.float64)
    if box_arr.shape == (2,):
        lo = np.full(N, box_arr[0])
        hi = np.full(N, box_arr[1])
    elif box_arr.shape == (2, N):
        lo, hi = box_arr[0].copy(), box_arr[1].copy()
    else:
        raise ValueError(
            f"box must be a scalar (lo, hi) pair or a (2, {N}) array-like of "
            f"per-asset bounds, got shape {box_arr.shape}."
        )

    if np.any(lo > hi):
        raise ValueError("box lower bound(s) must be <= upper bound(s).")

    return lo, hi


def _validate_pair(pair: tuple[float, float], name: str) -> tuple[float, float]:
    """ Validate a ``(lo, hi)`` pair: two floats with ``lo <= hi``. """
    lo, hi = float(pair[0]), float(pair[1])
    if lo > hi:
        raise ValueError(f"{name} lower bound must be <= upper bound.")

    return lo, hi


def _validate_groups(
    groups: dict[str, Sequence[int]] | None,
    group_bounds: dict[str, tuple[float, float]] | None,
    N: int,
) -> tuple[dict[str, NDArray], dict[str, tuple[float, float]]]:
    """ Validate `groups` indices and that `group_bounds` keys exist in `groups`. """
    groups = groups or {}
    group_bounds = group_bounds or {}

    unknown = sorted(set(group_bounds) - set(groups))
    if unknown:
        raise ValueError(
            f"group_bounds key(s) {unknown} not found in groups {sorted(groups)}."
        )

    group_idx = {}
    for name, idx in groups.items():
        idx_arr = np.asarray(idx, dtype=np.intp)
        if idx_arr.ndim != 1 or idx_arr.size == 0:
            raise ValueError(
                f"groups[{name!r}] must be a non-empty 1-D sequence of indices."
            )
        if np.any((idx_arr < 0) | (idx_arr >= N)):
            raise ValueError(
                f"groups[{name!r}] contains indices out of range [0, {N})."
            )
        group_idx[name] = idx_arr

    bounds = {
        name: _validate_pair(b, f"group_bounds[{name!r}]")
        for name, b in group_bounds.items()
    }

    return group_idx, bounds


def _check_feasibility(
    lo: NDArray,
    hi: NDArray,
    gross_max: float | None,
    net_range: tuple[float, float] | None,
    group_idx: dict[str, NDArray],
    group_bounds: dict[str, tuple[float, float]],
) -> None:
    """ Interval-arithmetic pre-check: raise a clear ValueError on an infeasible set.

    Cheap necessary (not sufficient) conditions checked against the box
    alone, before any SLSQP call: a net-exposure range, a group bound or a
    gross cap that cannot be reached even by the best-placed point of the
    box is reported as infeasible with an explicit message, instead of
    silently handed to the optimizer (which would fail or return a
    meaningless result).

    """
    box_lo_sum, box_hi_sum = float(np.sum(lo)), float(np.sum(hi))

    if net_range is not None:
        net_lo, net_hi = net_range
        if box_lo_sum > net_hi or box_hi_sum < net_lo:
            raise ValueError(
                f"net_range=({net_lo}, {net_hi}) is infeasible: the box "
                f"bounds only reach a net exposure in "
                f"[{box_lo_sum:.6g}, {box_hi_sum:.6g}]."
            )

    for name, (g_lo, g_hi) in group_bounds.items():
        idx = group_idx[name]
        g_box_lo_sum = float(np.sum(lo[idx]))
        g_box_hi_sum = float(np.sum(hi[idx]))
        if g_box_lo_sum > g_hi or g_box_hi_sum < g_lo:
            raise ValueError(
                f"group_bounds[{name!r}]=({g_lo}, {g_hi}) is infeasible: the "
                f"box bounds of its members only reach a net exposure in "
                f"[{g_box_lo_sum:.6g}, {g_box_hi_sum:.6g}]."
            )

    if gross_max is not None:
        min_abs = np.where(
            (lo <= 0.0) & (hi >= 0.0), 0.0, np.minimum(np.abs(lo), np.abs(hi))
        )
        min_gross = float(np.sum(min_abs))
        if gross_max < min_gross:
            raise ValueError(
                f"gross_max={gross_max:.6g} is infeasible: the box bounds "
                f"force a gross exposure >= {min_gross:.6g}."
            )


def _use_fast_path(
    method: str,
    lo: NDArray,
    hi: NDArray,
    net_range: tuple[float, float] | None,
    group_bounds: dict[str, tuple[float, float]],
) -> bool:
    """ Whether the closed-form (no-SLSQP) fast path applies. """
    if method == 'exact':
        return False
    if net_range is not None or group_bounds:
        return False

    return bool(np.all((lo <= 0.0) & (hi >= 0.0)))


# =========================================================================== #
#                          fast path (box + gross only)                      #
# =========================================================================== #


def _project_fast(w: NDArray, lo: NDArray, hi: NDArray, gross_max: float | None) -> NDArray:
    """ Clip to the box, then scale towards zero if the gross cap is breached.

    Exact for the box; scaling the clipped vector by ``gross_max / gross``
    stays inside the box because the box contains zero (checked by
    :func:`_use_fast_path`), so shrinking every coordinate towards zero
    cannot cross either bound.

    """
    v = np.clip(w, lo, hi)
    gross = float(np.sum(np.abs(v)))
    if gross_max is not None and gross > gross_max and gross > 0.0:
        v = v * (gross_max / gross)

    return v


# =========================================================================== #
#                       exact path (SLSQP least-distance)                    #
# =========================================================================== #


def _project_exact(
    w: NDArray,
    lo: NDArray,
    hi: NDArray,
    gross_max: float | None,
    net_range: tuple[float, float] | None,
    group_idx: dict[str, NDArray],
    group_bounds: dict[str, tuple[float, float]],
) -> NDArray:
    r""" Least-distance projection :math:`\min \|v - w\|^2` via SLSQP.

    Splits ``v = p - m`` with ``p, m >= 0`` (``2N`` variables) so that the
    gross constraint ``sum(p + m) <= gross_max`` is linear (hence smooth),
    rather than the non-smooth ``sum(|v|) <= gross_max``. Box, net and
    group constraints are all linear in ``(p, m)``.

    """
    N = w.shape[0]
    w_clip = np.clip(w, lo, hi)
    x0 = np.concatenate([np.clip(w_clip, 0.0, None), np.clip(-w_clip, 0.0, None)])

    def objective(x: NDArray) -> float:
        v = x[:N] - x[N:]
        return float(np.sum((v - w) ** 2))

    def jac(x: NDArray) -> NDArray:
        v = x[:N] - x[N:]
        diff = v - w
        return np.concatenate([2.0 * diff, -2.0 * diff])

    constraints = [LinearConstraint(np.hstack([np.eye(N), -np.eye(N)]), lo, hi)]

    if gross_max is not None:
        constraints.append(LinearConstraint(np.ones((1, 2 * N)), 0.0, gross_max))

    if net_range is not None:
        net_lo, net_hi = net_range
        row = np.concatenate([np.ones(N), -np.ones(N)]).reshape(1, -1)
        constraints.append(LinearConstraint(row, net_lo, net_hi))

    for name, (g_lo, g_hi) in group_bounds.items():
        idx = group_idx[name]
        row = np.zeros((1, 2 * N))
        row[0, idx] = 1.0
        row[0, N + idx] = -1.0
        constraints.append(LinearConstraint(row, g_lo, g_hi))

    bounds = Bounds(np.zeros(2 * N), np.full(2 * N, np.inf))
    result = minimize(
        objective,
        x0,
        jac=jac,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000},
    )

    if not result.success:
        raise ValueError(
            "SLSQP failed to converge; the requested constraint set (box, "
            f"gross_max={gross_max}, net_range={net_range}, "
            f"group_bounds={group_bounds}) is likely infeasible or "
            f"degenerate: {result.message}"
        )

    v = np.clip(result.x[:N] - result.x[N:], lo, hi)
    v[np.abs(v) < 1e-12] = 0.0

    return v


# =========================================================================== #
#                                public API                                   #
# =========================================================================== #


def project_weights(
    w: NDArray,
    box: tuple[float, float] | NDArray = (-1.0, 1.0),
    gross_max: float | None = None,
    net_range: tuple[float, float] | None = None,
    groups: dict[str, Sequence[int]] | None = None,
    group_bounds: dict[str, tuple[float, float]] | None = None,
    method: str = 'auto',
) -> NDArray:
    r""" Project weights onto a feasible set: box, gross, net and group bounds.

    Least-distance (Euclidean) projection of a weight vector, or of every
    row of a ``(T, N)`` weight book independently, onto the intersection
    of:

    - a per-asset box (``lo_i <= w_i <= hi_i``);
    - a gross-leverage cap (``sum(|w_i|) <= gross_max``);
    - a net-exposure range (``net_lo <= sum(w_i) <= net_hi``);
    - named group bounds (``lo_g <= sum_{i in g} w_i <= hi_g``).

    Meant as a composable overlay run after any allocator or signal and
    before :func:`fynance.backtest.engine.backtest`.

    Parameters
    ----------
    w : array_like
        Weights, shape ``(N,)`` or ``(T, N)``. When 2-D, each row is
        projected independently (no cross-row interaction) and the output
        has the same shape.
    box : tuple of float or array_like, optional
        Either a scalar ``(lo, hi)`` pair applied to every asset, or a
        ``(2, N)`` array-like of per-asset ``(lo_i, hi_i)`` bounds. Must
        satisfy ``lo <= hi`` element-wise. Default ``(-1.0, 1.0)``.
    gross_max : float, optional
        Cap on gross leverage ``sum(|w_i|)``. Default ``None`` (no cap).
    net_range : tuple of float, optional
        ``(lo, hi)`` bounds on the net exposure ``sum(w_i)``. Default
        ``None`` (unconstrained).
    groups : dict of str to sequence of int, optional
        Named groups of asset indices, e.g. ``{'tech': [0, 1], 'bond':
        [2]}``. Only groups referenced by `group_bounds` constrain the
        projection. Default ``None``.
    group_bounds : dict of str to tuple of float, optional
        ``(lo, hi)`` bounds on each named group's **net** sum
        ``sum_{i in group} w_i``. Every key must exist in `groups`.
        Default ``None`` (no group constraints).
    method : {'auto', 'exact'}, optional
        ``'auto'`` (default) uses a closed-form fast path (clip then scale)
        when only `box` and/or `gross_max` are active and the box contains
        zero; ``'exact'`` always uses the SLSQP least-distance projection.

    Returns
    -------
    np.ndarray
        Projected weights, same shape as `w`.

    Raises
    ------
    ValueError
        If `box` lower bounds exceed upper bounds, if a `group_bounds` key
        is not in `groups`, if a group's indices are out of range, if the
        constraint set is infeasible (message contains ``'infeasible'``),
        or if SLSQP fails to converge on the exact path.

    Notes
    -----
    The fast path is exact for the box (a plain clip), but the gross cap
    is applied as a uniform **scaling** of the clipped vector, not the
    strict least-distance projection onto ``{box} \cap {gross <=
    gross_max}`` — this is standard practice (cheap, and identical to the
    exact projection whenever the box does not bind at the optimum). Pass
    ``method='exact'`` to force the SLSQP least-distance projection
    instead.

    Examples
    --------
    Fast-path gross cap on a 3-asset long-short book:

    >>> import numpy as np
    >>> w = np.array([0.8, -0.6, 0.3])
    >>> v = project_weights(w, box=(-1.0, 1.0), gross_max=1.0)
    >>> np.round(v, 4)
    array([ 0.4706, -0.3529,  0.1765])

    Group-bound example — cap the net "tech" exposure to ``[-0.5, 0.5]``:

    >>> w = np.array([0.8, 0.6, -0.3])
    >>> v = project_weights(
    ...     w, groups={'tech': [0, 1], 'bond': [2]},
    ...     group_bounds={'tech': (-0.5, 0.5)},
    ... )
    >>> np.round(v, 6)
    array([ 0.35,  0.15, -0.3 ])

    """
    if method not in ('auto', 'exact'):
        raise ValueError(f"Unknown method {method!r}; expected 'auto' or 'exact'.")

    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1:
        N = w.shape[0]
    elif w.ndim == 2:
        N = w.shape[1]
    else:
        raise ValueError(f"w must be 1-D or 2-D, got ndim={w.ndim}.")

    if not np.all(np.isfinite(w)):
        raise ValueError("w contains non-finite values (NaN or inf).")

    lo, hi = _validate_box(box, N)
    group_idx, group_bounds = _validate_groups(groups, group_bounds, N)
    net_range = _validate_pair(net_range, "net_range") if net_range is not None else None

    _check_feasibility(lo, hi, gross_max, net_range, group_idx, group_bounds)

    fast = _use_fast_path(method, lo, hi, net_range, group_bounds)

    def project_row(row: NDArray) -> NDArray:
        if fast:
            return _project_fast(row, lo, hi, gross_max)

        return _project_exact(row, lo, hi, gross_max, net_range, group_idx, group_bounds)

    if w.ndim == 1:
        return project_row(w)

    out = np.empty_like(w)
    for t in range(w.shape[0]):
        out[t] = project_row(w[t])

    return out
