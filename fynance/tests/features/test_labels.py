#!/usr/bin/env python3
# coding: utf-8

""" Tests for the AFML labeling stack (triple-barrier, meta-labels, weights). """

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.features.labels import (
    LABEL_DTYPE,
    label_concurrency,
    meta_labels,
    triple_barrier,
    uniqueness_weights,
)

# --------------------------------------------------------------------------- #
#   triple_barrier -- deterministic paths                                     #
# --------------------------------------------------------------------------- #


def test_upper_barrier_touch_exact():
    prices = np.array([100., 101., 105., 101., 98., 97.])
    vol = np.full(6, 0.02)
    out = triple_barrier(prices, events=np.array([0]), pt=1.0, sl=1.0,
                          max_holding=4, vol=vol)
    assert out['t_in'][0] == 0
    assert out['t_out'][0] == 2
    assert out['label'][0] == 1
    assert out['ret'][0] == pytest.approx(0.05)


def test_lower_barrier_touch_exact():
    # Mirror image of the upper-touch path.
    prices = np.array([100., 99., 95., 99., 102., 103.])
    vol = np.full(6, 0.02)
    out = triple_barrier(prices, events=np.array([0]), pt=1.0, sl=1.0,
                          max_holding=4, vol=vol)
    assert out['t_in'][0] == 0
    assert out['t_out'][0] == 2
    assert out['label'][0] == -1
    assert out['ret'][0] == pytest.approx(-0.05)


def test_flat_path_resolves_at_vertical_bar():
    prices = np.array([100., 100.5, 100.2, 100.8, 100.1])
    vol = np.full(5, 1.0)  # barriers effectively unreachable
    out = triple_barrier(prices, events=np.array([0]), pt=1.0, sl=1.0,
                          max_holding=3, vol=vol)
    assert out['t_out'][0] == 3  # min(0 + 3, T - 1) = 3
    assert out['label'][0] == 0
    assert out['ret'][0] == pytest.approx(prices[3] / prices[0] - 1.0)


def test_pt_sl_asymmetry_honored():
    # Same downward path, only sl changes -> different resolution.
    prices = np.array([100., 99., 98.])
    vol = np.full(3, 0.01)

    tight = triple_barrier(prices, events=np.array([0]), pt=10.0, sl=1.0,
                            max_holding=2, vol=vol)
    assert tight['label'][0] == -1
    assert tight['t_out'][0] == 1

    loose = triple_barrier(prices, events=np.array([0]), pt=10.0, sl=3.0,
                            max_holding=2, vol=vol)
    assert loose['label'][0] == 0
    assert loose['t_out'][0] == 2


def test_vol_scaling_touch_vs_no_touch():
    prices = np.array([100., 102., 100.])
    v = 0.02

    touches = triple_barrier(prices, events=np.array([0]), pt=1.0, sl=1.0,
                              max_holding=2, vol=np.full(3, v))
    assert touches['label'][0] == 1
    assert touches['t_out'][0] == 1

    no_touch = triple_barrier(prices, events=np.array([0]), pt=1.0, sl=1.0,
                               max_holding=2, vol=np.full(3, 2 * v))
    assert no_touch['label'][0] == 0
    assert no_touch['t_out'][0] == 2


def test_default_events_and_vol_shape():
    prices = np.array([100., 101., 99., 102.])
    out = triple_barrier(prices)
    assert out.shape == (3,)  # np.arange(T - 1)
    assert np.array_equal(out['t_in'], np.arange(3))


def test_invalid_inputs_raise():
    prices = np.array([100., 101., 102.])

    with pytest.raises(ValueError):
        triple_barrier(prices, events=np.array([2]))  # T - 2 = 1, 2 is out of range

    with pytest.raises(ValueError):
        triple_barrier(prices, pt=0.0)

    with pytest.raises(ValueError):
        triple_barrier(prices, sl=-1.0)

    with pytest.raises(ValueError):
        triple_barrier(prices, max_holding=0)

    with pytest.raises(ValueError):
        triple_barrier(prices, vol=np.array([0.01, 0.02]))  # wrong shape

    with pytest.raises(ValueError):
        triple_barrier(np.array([100.]))  # T < 2


def test_structured_dtype_stable():
    prices = np.array([100., 101., 102.])
    out = triple_barrier(prices)
    assert out.dtype.names == ('t_in', 't_out', 'label', 'ret')
    assert out.dtype['t_in'] == np.int64
    assert out.dtype['t_out'] == np.int64
    assert out.dtype['label'] == np.int8
    assert out.dtype['ret'] == np.float64
    assert out.dtype == LABEL_DTYPE


# --------------------------------------------------------------------------- #
#   triple_barrier -- kernel parity vs a slow pure-Python reference            #
# --------------------------------------------------------------------------- #


def _slow_triple_barrier(prices, events, pt, sl, max_holding, scale):
    """ Pure-Python (no Numba) reference triple-barrier scan. """
    n = len(events)
    T = len(prices)
    t_in = np.empty(n, dtype=np.int64)
    t_out = np.empty(n, dtype=np.int64)
    label = np.empty(n, dtype=np.int8)
    ret = np.empty(n, dtype=np.float64)

    for k in range(n):
        i = int(events[k])
        p0 = prices[i]
        upper = pt * scale[k]
        lower = -sl * scale[k]
        vertical = min(i + max_holding, T - 1)

        lab = 0
        out_j = vertical
        for j in range(i + 1, vertical + 1):
            r = prices[j] / p0 - 1.0
            if r >= upper:
                lab = 1
                out_j = j
                break
            if r <= lower:
                lab = -1
                out_j = j
                break

        t_in[k] = i
        t_out[k] = out_j
        label[k] = lab
        ret[k] = prices[out_j] / p0 - 1.0

    return t_in, t_out, label, ret


def test_kernel_parity_seeded_gbm():
    rng = np.random.default_rng(42)
    T = 500
    prices = 100. * np.cumprod(1. + rng.standard_normal(T) * 0.01)

    # Causal trailing realized vol of simple returns (21-bar), so the vol
    # path through the kernel is exercised too, not just the constant default.
    simple_ret = np.empty(T)
    simple_ret[0] = 0.0
    simple_ret[1:] = prices[1:] / prices[:-1] - 1.0
    w = 21
    vol = np.empty(T)
    for t in range(T):
        lo = max(0, t - w + 1)
        vol[t] = simple_ret[lo:t + 1].std() if t > 0 else 1e-4
    vol[vol == 0] = 1e-4  # avoid degenerate zero-width barriers

    events = np.arange(T - 1)
    pt, sl, max_holding = 1.5, 0.8, 10
    scale = vol[events]

    fast = triple_barrier(prices, events=events, pt=pt, sl=sl,
                           max_holding=max_holding, vol=vol)
    slow_t_in, slow_t_out, slow_label, slow_ret = _slow_triple_barrier(
        prices, events, pt, sl, max_holding, scale,
    )

    assert np.array_equal(fast['t_in'], slow_t_in)
    assert np.array_equal(fast['t_out'], slow_t_out)
    assert np.array_equal(fast['label'], slow_label)
    assert np.array_equal(fast['ret'], slow_ret)
    # Sanity: all three label classes should appear over 500 bars.
    assert set(np.unique(fast['label'])) == {-1, 0, 1}


# --------------------------------------------------------------------------- #
#   meta_labels                                                               #
# --------------------------------------------------------------------------- #


def test_meta_labels_truth_table():
    sides = np.array([1, 1, 1, -1, -1, -1, 0, 0, 0])
    rets = np.array([0.1, -0.1, 0.0] * 3)
    labels = np.zeros(9, dtype=LABEL_DTYPE)
    labels['ret'] = rets

    expected = np.array([1., 0., 0., 0., 1., 0., 0., 0., 0.])
    assert np.array_equal(meta_labels(sides, labels), expected)


def test_meta_labels_length_mismatch_raises():
    labels = np.zeros(3, dtype=LABEL_DTYPE)
    with pytest.raises(ValueError):
        meta_labels(np.array([1, -1]), labels)


# --------------------------------------------------------------------------- #
#   label_concurrency                                                         #
# --------------------------------------------------------------------------- #


def test_label_concurrency_hand_checked_overlap():
    # Event 0 spans bars 0-2, event 1 spans bars 1-3.
    t_in = np.array([0, 1])
    t_out = np.array([2, 3])
    conc = label_concurrency(t_in, t_out, T=4)
    assert np.array_equal(conc, np.array([1, 2, 2, 1]))


def test_label_concurrency_length_mismatch_raises():
    with pytest.raises(ValueError):
        label_concurrency(np.array([0, 1]), np.array([1]), T=4)


def test_label_concurrency_out_of_range_raises():
    with pytest.raises(ValueError):
        label_concurrency(np.array([0]), np.array([4]), T=4)  # t_out == T
    with pytest.raises(ValueError):
        label_concurrency(np.array([2]), np.array([1]), T=4)  # t_in > t_out


# --------------------------------------------------------------------------- #
#   uniqueness_weights                                                        #
# --------------------------------------------------------------------------- #


def test_uniqueness_weights_disjoint_all_one():
    t_in = np.array([0, 2, 5])
    t_out = np.array([1, 3, 7])
    weights = uniqueness_weights(t_in, t_out, T=10)
    assert weights == pytest.approx(np.ones(3), rel=1e-12)


def test_uniqueness_weights_identical_events_equal_and_sum_to_n():
    t_in = np.array([5, 5, 5])
    t_out = np.array([8, 8, 8])
    weights = uniqueness_weights(t_in, t_out, T=10)
    assert weights[0] == pytest.approx(weights[1], rel=1e-12)
    assert weights[1] == pytest.approx(weights[2], rel=1e-12)
    assert weights.sum() == pytest.approx(3.0, rel=1e-12)


def test_uniqueness_weights_partial_overlap_matches_hand_computation():
    # Event 0/1 both span bars 0-1 (fully overlapping each other), event 2
    # spans bars 2-3 alone (disjoint from 0 and 1).
    t_in = np.array([0, 0, 2])
    t_out = np.array([1, 1, 3])
    weights = uniqueness_weights(t_in, t_out, T=4)
    assert weights == pytest.approx(np.array([0.75, 0.75, 1.5]), rel=1e-12)
    assert weights.sum() == pytest.approx(3.0, rel=1e-12)
