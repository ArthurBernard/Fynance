#!/usr/bin/env python3
# coding: utf-8

""" Tests for §5.6 robust-training utilities. """

import numpy as np
import pytest

from fynance.models.training import EarlyStopping, exp_sample_weights


def test_exp_sample_weights_values():
    assert np.allclose(exp_sample_weights(4, halflife=1), [0.125, 0.25, 0.5, 1.0])


def test_exp_sample_weights_recent_is_one_and_increasing():
    w = exp_sample_weights(50, halflife=10)
    assert w[-1] == 1.0
    assert np.all(np.diff(w) > 0)


def test_exp_sample_weights_bad_halflife():
    with pytest.raises(ValueError):
        exp_sample_weights(10, halflife=0)


def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=2, mode='max')
    results = [es.step(v) for v in [1.0, 0.9, 0.8]]
    assert results == [False, False, True]


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2, mode='max')
    for v in [1.0, 0.9, 1.5, 1.4]:
        stop = es.step(v)
    assert es.best == 1.5
    assert stop is False  # only one non-improving step since the 1.5 peak


def test_early_stopping_min_mode():
    es = EarlyStopping(patience=1, mode='min')
    assert [es.step(v) for v in [1.0, 0.5, 0.6]] == [False, False, True]


def test_early_stopping_bad_mode():
    with pytest.raises(ValueError):
        EarlyStopping(mode='nope')
