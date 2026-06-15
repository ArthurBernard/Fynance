#!/usr/bin/env python3
# coding: utf-8

""" Tests for time-ordered splits (no lookahead). """

# Third-party packages
import numpy as np

# Local packages
from fynance.data import train_test_split, walk_forward


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
