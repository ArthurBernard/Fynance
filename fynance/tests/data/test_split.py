#!/usr/bin/env python3
# coding: utf-8

""" Tests for time-ordered splits (no lookahead). """

# Built-in packages
import math

# Third-party packages
import numpy as np
import pytest

# Local packages
from fynance.data import combinatorial_purged_cv, train_test_split, walk_forward


def test_train_test_split_fraction():
    train, test = train_test_split(100, test_size=0.2)
    assert train[-1] < test[0]            # strictly ordered
    assert len(test) == 20
    assert len(train) == 80


def test_train_test_split_gap_embargo():
    train, test = train_test_split(100, test_size=10, gap=5)
    assert len(test) == 10
    # embargo of 5 between train end and test start
    assert test[0] - train[-1] == 1 + 5


def test_walk_forward_windows_ordered_and_disjoint():
    windows = list(walk_forward(100, train=40, test=10, step=10))
    assert len(windows) == 6  # t = 40,50,...,90
    for tr, te in windows:
        assert tr[-1] < te[0]                      # no leakage
        assert len(set(tr) & set(te)) == 0         # disjoint fold


def test_walk_forward_purge():
    windows = list(walk_forward(100, train=40, test=10, step=10, purge=3))
    for tr, te in windows:
        # purge removes 3 obs at the train/test boundary
        assert te[0] - tr[-1] == 1 + 3


def test_walk_forward_no_future_leak():
    # every train index must be strictly before its test window
    for tr, te in walk_forward(60, train=20, test=5, step=5):
        assert np.all(tr < te[0])


def test_walk_forward_rejects_purge_geq_train():
    # purge >= train would silently yield EMPTY train windows.
    with pytest.raises(ValueError, match="purge"):
        list(walk_forward(20, train=5, test=3, purge=5))
    with pytest.raises(ValueError, match="purge"):
        list(walk_forward(20, train=5, test=3, purge=6))


def test_walk_forward_rejects_nonpositive_train():
    with pytest.raises(ValueError, match="train"):
        list(walk_forward(20, train=0, test=3))


def test_train_test_split_fraction_one_is_count():
    # Documented: 1.0 is an absolute count (1), NOT the whole series.
    train, test = train_test_split(100, test_size=1.0)
    assert len(test) == 1
    assert len(train) == 99


def test_train_test_split_zero_is_empty_test():
    # Documented: 0.0 is the absolute count 0 -> empty test set.
    train, test = train_test_split(100, test_size=0.0)
    assert len(test) == 0
    assert len(train) == 100


def test_train_test_split_rejects_negative_test_size():
    # A negative integer would yield out-of-bounds train indices
    # (e.g. test_size=-3, n=10 -> split=13 -> arange(0, 13)); a negative
    # fraction would silently produce an empty test set.
    with pytest.raises(ValueError, match="test_size"):
        train_test_split(10, test_size=-3)
    with pytest.raises(ValueError, match="test_size"):
        train_test_split(10, test_size=-0.2)


def test_train_test_split_rejects_test_size_over_n():
    # A test count larger than n would leave a negative-length train set.
    with pytest.raises(ValueError, match="exceeds n"):
        train_test_split(10, test_size=11)


def test_walk_forward_rejects_nonpositive_step():
    # step <= 0 never advances t -> the while loop runs forever.
    # Use a tiny n and assert the ValueError is raised eagerly (the generator
    # validates step on the first __next__), so the loop is never entered.
    with pytest.raises(ValueError, match="step"):
        next(walk_forward(5, train=2, test=1, step=0))
    with pytest.raises(ValueError, match="step"):
        next(walk_forward(5, train=2, test=1, step=-1))


# --------------------------------------------------------------------------- #
#   combinatorial_purged_cv                                                   #
# --------------------------------------------------------------------------- #


def test_cpcv_split_count():
    folds = list(combinatorial_purged_cv(100, n_groups=6, n_test_groups=2))
    assert len(folds) == math.comb(6, 2)


def test_cpcv_each_group_appears_comb_times():
    n_groups, n_test_groups = 6, 2
    folds = list(combinatorial_purged_cv(120, n_groups=n_groups, n_test_groups=n_test_groups))

    # sizes as equal as possible -> 120 / 6 = 20 per group, so group g owns
    # indices [20*g, 20*g + 20).
    group_size = 120 // n_groups
    counts = np.zeros(n_groups, dtype=np.int64)
    for _, test_idx in folds:
        groups_in_test = {int(i) // group_size for i in test_idx}
        assert len(groups_in_test) == n_test_groups  # groups are never split
        for g in groups_in_test:
            counts[g] += 1

    assert np.all(counts == math.comb(n_groups - 1, n_test_groups - 1))


def test_cpcv_train_test_disjoint():
    for train_idx, test_idx in combinatorial_purged_cv(100, n_groups=6, n_test_groups=2, purge=3, embargo=2):
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0


def test_cpcv_purge_removes_boundary_bars():
    # Hand-checked: T=20, 4 groups (size 5 each) -> group 1 = [5, 10). With
    # purge=2, train drops bars {3, 4} just before and {10, 11} just after.
    folds = list(combinatorial_purged_cv(20, n_groups=4, n_test_groups=1, purge=2))
    train_idx, test_idx = folds[1]  # combo (1,) -> test group [5, 10)
    assert test_idx.tolist() == [5, 6, 7, 8, 9]
    assert train_idx.tolist() == [0, 1, 2, 12, 13, 14, 15, 16, 17, 18, 19]
    assert 3 not in train_idx and 4 not in train_idx
    assert 10 not in train_idx and 11 not in train_idx


def test_cpcv_embargo_removes_post_test_bars():
    # Same layout, purge=0 this time: only the embargo trims the 2 bars
    # {10, 11} immediately after the test group's end; nothing before it.
    folds = list(combinatorial_purged_cv(20, n_groups=4, n_test_groups=1, purge=0, embargo=2))
    train_idx, test_idx = folds[1]
    assert test_idx.tolist() == [5, 6, 7, 8, 9]
    assert train_idx.tolist() == [0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19]
    assert 10 not in train_idx and 11 not in train_idx


def test_cpcv_indices_in_range_sorted_unique():
    for train_idx, test_idx in combinatorial_purged_cv(50, n_groups=5, n_test_groups=2, purge=2, embargo=1):
        for idx in (train_idx, test_idx):
            assert idx.dtype == np.int64
            assert np.all(idx >= 0) and np.all(idx < 50)
            assert np.all(np.diff(idx) > 0)  # strictly increasing -> sorted & unique


def test_cpcv_rejects_bad_n_test_groups():
    with pytest.raises(ValueError, match="n_test_groups"):
        list(combinatorial_purged_cv(50, n_groups=5, n_test_groups=0))
    with pytest.raises(ValueError, match="n_test_groups"):
        list(combinatorial_purged_cv(50, n_groups=5, n_test_groups=5))
    with pytest.raises(ValueError, match="n_test_groups"):
        list(combinatorial_purged_cv(50, n_groups=5, n_test_groups=6))


def test_cpcv_rejects_n_groups_over_T():
    with pytest.raises(ValueError, match="n_groups"):
        list(combinatorial_purged_cv(4, n_groups=5, n_test_groups=1))


@pytest.mark.parametrize("h", [0, 3])
def test_cpcv_purge_no_train_within_h_of_test_boundary(h):
    # Property: with purge >= h, no train index lies within h bars before a
    # test block start or after a test block end (no embargo confound: 0).
    T, n_groups, n_test_groups = 60, 6, 2
    group_size = T // n_groups
    for train_idx, test_idx in combinatorial_purged_cv(
        T, n_groups=n_groups, n_test_groups=n_test_groups, purge=h, embargo=0,
    ):
        # Recover contiguous test blocks' [start, end) boundaries from test_idx.
        test_groups = sorted({int(i) // group_size for i in test_idx})
        boundaries = [(g * group_size, (g + 1) * group_size) for g in test_groups]

        train_set = set(train_idx.tolist())
        for start, end in boundaries:
            for offset in range(1, h + 1):
                assert start - offset not in train_set
                assert end + offset - 1 not in train_set


def test_cpcv_embargo_stacks_on_purge():
    # Regression: embargo must be applied *beyond* the post-test purge, not
    # overlap it. With purge=2 and embargo=2, the post-test exclusion after a
    # test group ending at bar e must span [e, e+4), not [e, e+2).
    folds = list(
        combinatorial_purged_cv(20, n_groups=4, n_test_groups=1, purge=2, embargo=2)
    )
    # Groups are [0,5),[5,10),[10,15),[15,20); pick the combo whose test is
    # group 1 = [5, 10) so both boundaries sit inside the series.
    train, test = next((tr, te) for tr, te in folds if te[0] == 5)
    assert test.tolist() == [5, 6, 7, 8, 9]
    # bars 10,11 purged; bars 12,13 embargoed -> none may be in train.
    for leaked in (10, 11, 12, 13):
        assert leaked not in train
    assert 14 in train  # first bar past purge+embargo is back in train
