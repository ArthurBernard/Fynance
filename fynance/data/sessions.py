#!/usr/bin/env python3
# coding: utf-8

""" Vendor-agnostic intraday trading-session utilities.

Deterministic, dependency-free tools to tag epoch-second timestamps with an
exchange's regular trading hours: a boolean mask, a per-bar running session
id, first/last-bar bounds per session, and a helper to split an array into
one chunk per session.

**Scope**: timestamps are plain ``int64``/``float64`` NumPy arrays holding
epoch seconds (UTC) -- no ``pandas``, ``pytz`` or ``exchange_calendars``
dependency. The exchange clock is modelled as a single fixed ``utc_offset``
(hours). **Daylight-saving time is explicitly out of scope**: a fixed offset
cannot represent an exchange that observes DST (its local open/close would
drift by an hour twice a year). Users who need DST-correct sessions should
bring ``exchange_calendars`` (or similar) themselves and pre-shift their
timestamps, or call these functions twice with the two seasonal offsets. This
module is the deterministic, zero-dependency tool for the common 80% case: a
fixed-offset exchange (or an already DST-adjusted timestamp array).

"""

from __future__ import annotations

# Built-in packages
# Third-party packages
import numpy as np
from numpy.typing import NDArray

__all__ = ['session_mask', 'session_id', 'session_bounds', 'split_sessions']

_SECONDS_PER_DAY = 86400
# 1970-01-01 (epoch day 0) is a Thursday; datetime.date.weekday() convention
# (Monday=0 ... Sunday=6), so weekday = (epoch_day + 3) % 7.
_EPOCH_DAY_0_WEEKDAY_OFFSET = 3
_WEEKEND_CUTOFF = 5  # weekday >= 5 -> Saturday/Sunday


def _parse_hhmm(value: str) -> int:
    """ Parse an ``"HH:MM"`` string into seconds since local midnight.

    Parameters
    ----------
    value : str
        Time of day, e.g. ``"09:30"``.

    Returns
    -------
    int
        Seconds since local midnight, in ``[0, 86400)``.

    Raises
    ------
    ValueError
        If ``value`` is not a well-formed ``"HH:MM"`` string with
        ``0 <= HH < 24`` and ``0 <= MM < 60``.

    """
    parts = value.split(":")

    if len(parts) != 2:

        raise ValueError(f"invalid HH:MM time string: {value!r}")

    try:
        hour, minute = int(parts[0]), int(parts[1])

    except ValueError as exc:

        raise ValueError(f"invalid HH:MM time string: {value!r}") from exc

    if not (0 <= hour < 24 and 0 <= minute < 60):

        raise ValueError(f"invalid HH:MM time string: {value!r}")

    return hour * 3600 + minute * 60


def _check_timestamps(ts: NDArray[np.int64] | NDArray[np.float64]) -> NDArray[np.float64]:
    """ Validate ``ts`` and return it as a 1-D ``float64`` array.

    Raises
    ------
    ValueError
        If ``ts`` is not 1-D, or is not non-decreasing.

    """
    arr = np.asarray(ts)

    if arr.ndim != 1:

        raise ValueError(f"ts must be a 1-D array, got shape {arr.shape}")

    if arr.size > 1 and np.any(np.diff(arr) < 0):

        raise ValueError("ts must be non-decreasing")

    return arr.astype(np.float64)


def _session_components(
    ts: NDArray[np.int64] | NDArray[np.float64],
    open: str,
    close: str,
    utc_offset: float,
    weekdays_only: bool,
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """ Shared core: per-bar in-session flag and the session's opening day.

    ``open_day`` is the local epoch-day number (``local_seconds // 86400``,
    epoch day 0 = Thursday 1970-01-01) on which the bar's session OPENED --
    for a regular (same-day) session this is just the bar's own calendar day,
    but for an overnight session (``close < open``) the post-midnight half of
    the session still carries the PREVIOUS day's ``open_day``. This lets
    :func:`session_id` group an overnight session's bars under one id, and
    lets ``weekdays_only`` decide inclusion from the day the session opened
    rather than the calendar day a given bar happens to fall on.

    """
    arr = _check_timestamps(ts)
    open_sod = _parse_hhmm(open)
    close_sod = _parse_hhmm(close)

    # Pure (float64, but integer-valued) arithmetic hot path: days = ts // 86400.
    local = arr + utc_offset * 3600.0
    days = np.floor(local / _SECONDS_PER_DAY).astype(np.int64)
    seconds_of_day = local - days * float(_SECONDS_PER_DAY)

    if open_sod < close_sod:
        # Regular, same-day session: [open, close).
        in_session = (seconds_of_day >= open_sod) & (seconds_of_day < close_sod)
        open_day = days

    else:
        # Overnight session (close <= open, e.g. 18:00 -> 17:00): a bar either
        # falls in the evening half [open, 24:00) of the day it opened, or in
        # the morning half [00:00, close) that belongs to the PREVIOUS day's
        # session. open_sod == close_sod is the degenerate 24h-session case
        # (every bar matches one of the two halves).
        evening = seconds_of_day >= open_sod
        morning = seconds_of_day < close_sod
        in_session = evening | morning
        open_day = np.where(evening, days, days - 1)

    if weekdays_only:
        weekday = (open_day + _EPOCH_DAY_0_WEEKDAY_OFFSET) % 7
        in_session = in_session & (weekday < _WEEKEND_CUTOFF)

    return in_session, open_day


def session_mask(
    ts: NDArray[np.int64] | NDArray[np.float64],
    open: str = "09:30",
    close: str = "16:00",
    utc_offset: float = 0.0,
    weekdays_only: bool = True,
) -> NDArray[np.bool_]:
    """ Flag timestamps that fall inside a trading session.

    Parameters
    ----------
    ts : ndarray of int64 or float64
        Non-decreasing epoch-second (UTC) timestamps, shape ``(n,)``.
    open, close : str, default "09:30", "16:00"
        Local session open/close, ``"HH:MM"``. ``close < open`` (or
        ``close == open``) models an overnight session, e.g. ``"18:00"`` ->
        ``"17:00"``.
    utc_offset : float, default 0.0
        Fixed exchange-clock offset from UTC, in hours (e.g. ``-5.0`` for
        NYSE standard time). A **single** fixed offset -- DST is out of
        scope, see the module docstring.
    weekdays_only : bool, default True
        Exclude sessions that OPEN on a Saturday/Sunday (local calendar day
        of the session's open, not of every individual bar -- see
        :func:`session_id`).

    Returns
    -------
    ndarray of bool
        ``True`` where ``ts`` falls in ``[open, close)`` local time (and, if
        ``weekdays_only``, the session opened on a weekday).

    Raises
    ------
    ValueError
        If ``open``/``close`` are not well-formed ``"HH:MM"`` strings, ``ts``
        is not 1-D, or ``ts`` is not non-decreasing.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.data.sessions import session_mask
    >>> ts = np.array([9 * 3600, 10 * 3600, 17 * 3600])  # 09:00, 10:00, 17:00 UTC
    >>> session_mask(ts, open="09:30", close="16:00")
    array([False,  True, False])

    """
    in_session, _ = _session_components(ts, open, close, utc_offset, weekdays_only)

    return in_session


def session_id(
    ts: NDArray[np.int64] | NDArray[np.float64],
    open: str = "09:30",
    close: str = "16:00",
    utc_offset: float = 0.0,
    weekdays_only: bool = True,
) -> NDArray[np.int64]:
    """ Label each bar with a running trading-session id.

    Bars outside a session get ``-1``. Bars inside a session get a
    0-indexed, strictly increasing id shared by every bar of that session
    (including across a data gap, e.g. a missing lunch bar): the id
    increments whenever the session's opening day (see
    :func:`_session_components`) changes from the previous in-session bar --
    not merely when an array index transition occurs -- so it is robust to
    ``ts`` that only contains in-session samples (no off-hours rows to mark
    a boundary with ``False``).

    Parameters
    ----------
    ts : ndarray of int64 or float64
        Non-decreasing epoch-second (UTC) timestamps, shape ``(n,)``.
    open, close, utc_offset, weekdays_only
        See :func:`session_mask`.

    Returns
    -------
    ndarray of int64
        Shape ``(n,)``; ``-1`` outside sessions, else a 0-indexed running
        session id.

    Raises
    ------
    ValueError
        Same as :func:`session_mask`.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.data.sessions import session_id
    >>> ts = np.array([9 * 3600, 10 * 3600, 17 * 3600, 33 * 3600, 34 * 3600])
    >>> session_id(ts, open="09:30", close="16:00")
    array([-1,  0, -1, -1,  1])

    """
    in_session, open_day = _session_components(ts, open, close, utc_offset, weekdays_only)
    ids = np.full(in_session.shape, -1, dtype=np.int64)

    if not np.any(in_session):

        return ids

    session_days = open_day[in_session]
    new_session = np.empty(session_days.shape, dtype=bool)
    new_session[0] = True
    new_session[1:] = session_days[1:] != session_days[:-1]
    ids[in_session] = np.cumsum(new_session) - 1

    return ids


def session_bounds(
    ts: NDArray[np.int64] | NDArray[np.float64],
    open: str = "09:30",
    close: str = "16:00",
    utc_offset: float = 0.0,
    weekdays_only: bool = True,
) -> NDArray[np.int64]:
    """ First/last bar index of each trading session.

    Parameters
    ----------
    ts : ndarray of int64 or float64
        Non-decreasing epoch-second (UTC) timestamps, shape ``(n,)``.
    open, close, utc_offset, weekdays_only
        See :func:`session_mask`.

    Returns
    -------
    ndarray of int64
        Shape ``(n_sessions, 2)``; row ``k`` is
        ``(first_index, last_index)`` (both inclusive, into ``ts``) of
        session ``k``. Empty (``shape (0, 2)``) if no bar falls in a
        session.

    Raises
    ------
    ValueError
        Same as :func:`session_mask`.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.data.sessions import session_bounds
    >>> ts = np.array([9 * 3600, 10 * 3600, 17 * 3600, 33 * 3600, 34 * 3600])
    >>> session_bounds(ts, open="09:30", close="16:00")
    array([[1, 1],
           [4, 4]])

    """
    ids = session_id(ts, open=open, close=close, utc_offset=utc_offset, weekdays_only=weekdays_only)
    in_session = ids >= 0

    if not np.any(in_session):

        return np.empty((0, 2), dtype=np.int64)

    index = np.nonzero(in_session)[0]
    valid_ids = ids[in_session]
    n_sessions = int(valid_ids[-1]) + 1
    boundaries = np.arange(n_sessions)
    first = index[np.searchsorted(valid_ids, boundaries, side="left")]
    last = index[np.searchsorted(valid_ids, boundaries, side="right") - 1]

    return np.stack([first, last], axis=1)


def split_sessions(
    X: NDArray,
    ts: NDArray[np.int64] | NDArray[np.float64],
    open: str = "09:30",
    close: str = "16:00",
    utc_offset: float = 0.0,
    weekdays_only: bool = True,
) -> list[NDArray]:
    """ Split ``X`` into one chunk per trading session.

    Parameters
    ----------
    X : ndarray
        Array aligned with ``ts`` along its first axis.
    ts : ndarray of int64 or float64
        Non-decreasing epoch-second (UTC) timestamps, shape ``(n,)``.
    open, close, utc_offset, weekdays_only
        See :func:`session_mask`.

    Returns
    -------
    list of ndarray
        One slice of ``X`` per session, in session order. Bars outside every
        session (``session_id == -1``) are skipped, so concatenating the
        result along axis 0 reproduces ``X[session_mask(ts, ...)]``.

    Raises
    ------
    ValueError
        Same as :func:`session_mask`.

    Examples
    --------
    >>> import numpy as np
    >>> from fynance.data.sessions import split_sessions
    >>> ts = np.array([9 * 3600, 10 * 3600, 17 * 3600, 33 * 3600, 34 * 3600])
    >>> X = np.arange(5)
    >>> split_sessions(X, ts, open="09:30", close="16:00")
    [array([1]), array([4])]

    """
    X = np.asarray(X)
    ids = session_id(ts, open=open, close=close, utc_offset=utc_offset, weekdays_only=weekdays_only)
    in_session = ids >= 0

    if not np.any(in_session):

        return []

    index = np.nonzero(in_session)[0]
    valid_ids = ids[in_session]
    n_sessions = int(valid_ids[-1]) + 1
    split_points = np.searchsorted(valid_ids, np.arange(1, n_sessions))
    groups = np.split(index, split_points)

    return [X[g] for g in groups]
