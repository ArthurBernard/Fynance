#!/usr/bin/env python3
# coding: utf-8

""" Tests for vendor-agnostic intraday trading-session utilities. """

# Built-in packages
import datetime

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.data import session_bounds, session_id, session_mask, split_sessions

UTC = datetime.timezone.utc


def _utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0,
) -> float:
    """ Build an epoch-second timestamp (UTC) from calendar fields. """
    return datetime.datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp()


def test_regular_rth_day_390_one_minute_bars():
    # Tuesday 2024-01-02, a full 24h grid of 1-min bars.
    start = _utc(2024, 1, 2)
    ts = np.array([start + i * 60 for i in range(24 * 60)])
    mask = session_mask(ts, open="09:30", close="16:00", utc_offset=0.0)
    assert mask.sum() == 390

    in_session = np.nonzero(mask)[0]
    first_dt = datetime.datetime(2024, 1, 2, tzinfo=UTC) + datetime.timedelta(minutes=int(in_session[0]))
    last_dt = datetime.datetime(2024, 1, 2, tzinfo=UTC) + datetime.timedelta(minutes=int(in_session[-1]))
    assert (first_dt.hour, first_dt.minute) == (9, 30)
    assert (last_dt.hour, last_dt.minute) == (15, 59)


def test_boundary_open_inclusive_close_exclusive():
    open_ts = _utc(2024, 1, 2, 9, 30)
    close_ts = _utc(2024, 1, 2, 16, 0)
    ts = np.array([open_ts - 60, open_ts, close_ts - 60, close_ts])
    mask = session_mask(ts, open="09:30", close="16:00")
    assert list(mask) == [False, True, True, False]


def test_weekend_excluded():
    # 2024-01-06/07 are a Saturday/Sunday (2024-01-01 was a Monday).
    saturday = _utc(2024, 1, 6, 10, 0)
    sunday = _utc(2024, 1, 7, 10, 0)
    ts = np.array([saturday, sunday])

    mask = session_mask(ts, open="09:30", close="16:00", weekdays_only=True)
    assert not mask.any()

    mask_all_days = session_mask(ts, open="09:30", close="16:00", weekdays_only=False)
    assert mask_all_days.all()


def test_utc_offset_shift():
    # 14:30 UTC == 09:30 local at utc_offset=-5 -> included (session open).
    assert session_mask(
        np.array([_utc(2024, 1, 2, 14, 30)]), open="09:30", close="16:00", utc_offset=-5.0,
    )[0]
    # 21:00 UTC == 16:00 local at utc_offset=-5 -> excluded (close is exclusive).
    assert not session_mask(
        np.array([_utc(2024, 1, 2, 21, 0)]), open="09:30", close="16:00", utc_offset=-5.0,
    )[0]
    # 13:00 UTC == 08:00 local at utc_offset=-5 -> excluded (before open).
    assert not session_mask(
        np.array([_utc(2024, 1, 2, 13, 0)]), open="09:30", close="16:00", utc_offset=-5.0,
    )[0]


def test_overnight_session_spans_midnight_single_id():
    # Tuesday 18:00 -> Wednesday 17:00: 23h of 1-min bars, one session.
    start = _utc(2024, 1, 2, 18, 0)
    ts = np.array([start + i * 60 for i in range(1380)])

    mask = session_mask(ts, open="18:00", close="17:00", weekdays_only=False)
    assert mask.sum() == 1380
    assert mask.all()

    ids = session_id(ts, open="18:00", close="17:00", weekdays_only=False)
    assert (ids == 0).all()
    # explicitly check the id is unchanged straddling the midnight boundary
    # (bar 359 == 23:59 Tue, bar 360 == 00:00 Wed).
    assert ids[359] == ids[360] == 0


def test_overnight_session_belongs_to_day_it_opens():
    # Opens Friday 18:00 (weekday): stays included through the Saturday-dated
    # post-midnight half -- the session belongs to the day it OPENED.
    start_fri = _utc(2024, 1, 5, 18, 0)  # 2024-01-05 is a Friday
    ts_fri = np.array([start_fri + i * 60 for i in range(1380)])
    assert session_mask(ts_fri, open="18:00", close="17:00", weekdays_only=True).all()

    # Opens Saturday 18:00 (weekend): excluded entirely, even though it bleeds
    # into the Sunday-dated post-midnight half.
    start_sat = _utc(2024, 1, 6, 18, 0)
    ts_sat = np.array([start_sat + i * 60 for i in range(1380)])
    assert not session_mask(ts_sat, open="18:00", close="17:00", weekdays_only=True).any()


def test_session_id_negative_one_gaps_and_strict_increments():
    day1 = datetime.datetime(2024, 1, 2, tzinfo=UTC)  # Tuesday
    day2 = datetime.datetime(2024, 1, 3, tzinfo=UTC)  # Wednesday
    ts = np.array([
        (day1 + datetime.timedelta(hours=8)).timestamp(),   # before open
        (day1 + datetime.timedelta(hours=10)).timestamp(),  # session 0
        (day1 + datetime.timedelta(hours=12)).timestamp(),  # session 0
        (day1 + datetime.timedelta(hours=20)).timestamp(),  # after close
        (day2 + datetime.timedelta(hours=10)).timestamp(),  # session 1
        (day2 + datetime.timedelta(hours=11)).timestamp(),  # session 1
    ])
    ids = session_id(ts, open="09:30", close="16:00")
    assert list(ids) == [-1, 0, 0, -1, 1, 1]


def test_split_sessions_concatenation_equals_masked_subset():
    start = _utc(2024, 1, 2)
    ts = np.array([start + i * 3600 for i in range(48)])  # 2 days, hourly
    X = np.arange(48)

    mask = session_mask(ts, open="09:30", close="16:00")
    chunks = split_sessions(X, ts, open="09:30", close="16:00")

    assert np.array_equal(np.concatenate(chunks), X[mask])
    # skips -1 (out-of-session) rows: chunk count == number of sessions
    assert len(chunks) == len(session_bounds(ts, open="09:30", close="16:00"))


def test_session_bounds_correctness():
    start = _utc(2024, 1, 2)
    ts = np.array([start + i * 3600 for i in range(48)])  # 2 days, hourly

    bounds = session_bounds(ts, open="09:30", close="16:00")
    ids = session_id(ts, open="09:30", close="16:00")

    assert bounds.shape == (2, 2)
    for k, (first, last) in enumerate(bounds):
        positions = np.nonzero(ids == k)[0]
        assert first == positions[0]
        assert last == positions[-1]


def test_session_bounds_and_split_sessions_empty_when_no_session():
    ts = np.array([_utc(2024, 1, 6, 10, 0)])  # Saturday only

    bounds = session_bounds(ts, open="09:30", close="16:00")
    assert bounds.shape == (0, 2)

    chunks = split_sessions(np.zeros(1), ts, open="09:30", close="16:00")
    assert chunks == []


def test_invalid_hhmm_raises():
    with pytest.raises(ValueError):
        session_mask(np.array([0.0]), open="9:xx")
    with pytest.raises(ValueError):
        session_mask(np.array([0.0]), open="25:00")
    with pytest.raises(ValueError):
        session_mask(np.array([0.0]), close="12:60")
    with pytest.raises(ValueError):
        session_mask(np.array([0.0]), open="09:30:00")


def test_non_decreasing_ts_raises():
    ts = np.array([10.0, 5.0, 20.0])
    with pytest.raises(ValueError, match="non-decreasing"):
        session_mask(ts)


def test_int64_and_float64_inputs_agree():
    ts_int = np.array([9 * 3600 + 1800, 10 * 3600], dtype=np.int64)
    ts_float = ts_int.astype(np.float64)
    assert np.array_equal(session_mask(ts_int), session_mask(ts_float))


def test_weekday_convention_golden_sweep_vs_datetime():
    # Golden-test the (epoch_day + 3) % 7 weekday convention against Python's
    # own datetime.weekday() over a ~750-day sweep, sampling local noon so
    # every bar sits inside an (almost) full-day session.
    n_days = 750
    epoch_0 = datetime.date(1970, 1, 1)
    ts = np.array([d * 86400 + 12 * 3600 for d in range(n_days)], dtype=np.int64)

    mask = session_mask(ts, open="00:00", close="23:59", utc_offset=0.0, weekdays_only=True)

    for d in range(n_days):
        expected_weekday = (epoch_0 + datetime.timedelta(days=d)).weekday()
        assert mask[d] == (expected_weekday < 5), f"day {d} mismatch"
