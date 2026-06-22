#!/usr/bin/env python3
# coding: utf-8

""" Tests for the signal mappers and pipeline. """

# Third-party packages
import numpy as np

# Local packages
from fynance.core import SignalModel
from fynance.signal import (
    SignalPipeline,
    deadband,
    ema_smooth,
    min_hold,
    rank,
    sign,
    threshold,
    vol_target_position,
)


def _turnover(pos):
    return float(np.abs(np.diff(np.asarray(pos, dtype=float), prepend=0.0)).sum())


def test_sign():
    assert np.array_equal(sign(np.array([0.3, -0.1, 0.0])), [1.0, -1.0, 0.0])


def test_threshold_dead_band():
    out = threshold(np.array([0.5, 0.05, -0.5]), long=0.1, short=-0.1)
    assert np.array_equal(out, [1.0, 0.0, -1.0])


def test_rank_long_short_dollar_neutral():
    pred = np.array([[1.0, 2.0, 3.0, 4.0]])
    w = rank(pred, top=1, bottom=1)
    assert np.isclose(w.sum(), 0.0)          # dollar-neutral
    assert w[0, 3] == 1.0                     # highest -> long
    assert w[0, 0] == -1.0                    # lowest -> short
    assert w[0, 1] == 0.0 and w[0, 2] == 0.0


def test_rank_requires_2d():
    import pytest
    with pytest.raises(ValueError):
        rank(np.array([1.0, 2.0, 3.0]), top=1, bottom=1)


def test_rank_rejects_overlapping_legs():
    # top + bottom > n_assets would let the long leg overwrite the short leg,
    # silently producing a non-dollar-neutral book (sum 0.667 in the report).
    import pytest
    with pytest.raises(ValueError):
        rank(np.array([[1.0, 2.0, 3.0, 4.0]]), top=3, bottom=3)


def test_rank_rejects_negative_legs():
    import pytest
    for top, bottom in ((-1, 1), (1, -1)):
        with pytest.raises(ValueError):
            rank(np.array([[1.0, 2.0, 3.0, 4.0]]), top=top, bottom=bottom)


def test_rank_dollar_neutral_when_legs_fit():
    # Any valid split (top + bottom <= n_assets) must be exactly dollar-neutral.
    pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                     [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    for top, bottom in ((1, 1), (2, 2), (3, 3), (2, 1), (1, 4)):
        w = rank(pred, top=top, bottom=bottom)
        assert np.allclose(w.sum(axis=1), 0.0)        # row-wise dollar-neutral
        assert np.allclose(w[w > 0].sum(), top * w.shape[0] * (1.0 / top))


def test_vol_target_position_is_causal():
    rng = np.random.default_rng(0)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 100))
    signal = np.ones(100)
    pos = vol_target_position(signal, prices, w=10)
    assert pos.shape == (100,)
    # perturbing a future price must not change earlier positions
    p2 = prices.copy()
    p2[60:] *= 1.5
    pos2 = vol_target_position(signal, p2, w=10)
    assert np.allclose(pos[:60], pos2[:60], equal_nan=True)


def test_signal_pipeline():
    class DummyModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.asarray(X).reshape(-1) - 0.5

    pipe = SignalPipeline(DummyModel(), sign)
    pipe.fit(np.zeros((3, 1)), np.zeros(3))
    pos = pipe.predict_position(np.array([0.9, 0.1, 0.5]))
    assert np.array_equal(pos, [1.0, -1.0, 0.0])


def test_dummy_model_conforms_to_signalmodel():
    class DummyModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    assert isinstance(DummyModel(), SignalModel)


# --- anti-churn mappers -----------------------------------------------------

def test_ema_smooth_cuts_turnover_and_is_causal():
    flips = np.array([1.0, -1.0] * 50)               # churns every bar
    smoothed = ema_smooth(flips, alpha=0.2)
    assert _turnover(smoothed) < _turnover(flips)     # turnover reduced
    # causal: a future value cannot change an earlier output
    other = flips.copy()
    other[60:] = 0.0
    assert np.allclose(ema_smooth(flips, 0.2)[:60], ema_smooth(other, 0.2)[:60])


def test_ema_smooth_alpha_one_is_identity():
    p = np.array([0.3, -0.7, 1.0, -0.2])
    assert np.allclose(ema_smooth(p, alpha=1.0), p)


def test_ema_smooth_rejects_bad_alpha():
    import pytest
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ema_smooth(np.zeros(3), alpha=bad)


def test_deadband_holds_through_small_moves():
    out = deadband(np.array([0.0, 0.05, 0.5, 0.55, -0.4]), band=0.2)
    assert np.allclose(out, [0.0, 0.0, 0.5, 0.5, -0.4])


def test_deadband_cuts_turnover():
    rng = np.random.default_rng(0)
    noisy = rng.normal(0, 0.05, 200)                 # jitter around 0
    assert _turnover(deadband(noisy, band=0.2)) < _turnover(noisy)


def test_deadband_is_causal():
    p = np.array([0.0, 0.5, 0.55, 0.9, -0.4])
    other = p.copy()
    other[3:] = 5.0
    assert np.allclose(deadband(p, 0.2)[:3], deadband(other, 0.2)[:3])


def test_min_hold_enforces_dwell():
    out = min_hold(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]), hold=3)
    assert np.allclose(out, [1.0, 1.0, 1.0, -1.0, -1.0, -1.0])


def test_min_hold_noop_when_hold_leq_one():
    p = np.array([1.0, -1.0, 1.0])
    assert np.allclose(min_hold(p, hold=1), p)


def test_min_hold_is_causal():
    p = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    other = p.copy()
    other[4:] = -1.0
    assert np.allclose(min_hold(p, 3)[:4], min_hold(other, 3)[:4])
