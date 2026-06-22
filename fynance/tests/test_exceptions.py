#!/usr/bin/env python3
# coding: utf-8

""" Tests for :class:`fynance._exceptions.ArraySizeError`. """

# Third-party packages
import pytest

# Local packages
from fynance._exceptions import ArraySizeError


def test_message_size_only():
    err = ArraySizeError(3)
    assert str(err) == "array of size 3 is not allowed"


def test_message_with_axis():
    err = ArraySizeError(3, axis=1)
    assert str(err) == "array of size 3 in axis 1 is not allowed"


def test_message_with_min_size():
    err = ArraySizeError(3, min_size=5)
    assert str(err) == "array of size 3 is not allowed, minimum size is 5"


def test_message_with_axis_and_min_size():
    err = ArraySizeError(3, axis=0, min_size=5)
    assert str(err) == (
        "array of size 3 in axis 0 is not allowed, minimum size is 5"
    )


def test_message_with_prefix():
    err = ArraySizeError(3, msg_prefix="feature matrix")
    assert str(err) == "feature matrix: array of size 3 is not allowed"


def test_message_with_all_fields():
    err = ArraySizeError(3, axis=1, min_size=5, msg_prefix="X")
    assert str(err) == (
        "X: array of size 3 in axis 1 is not allowed, minimum size is 5"
    )


def test_dual_inheritance_valueerror_and_indexerror():
    # ArraySizeError is both a ValueError and an IndexError, so callers can
    # catch it under either contract.
    err = ArraySizeError(0)
    assert isinstance(err, ValueError)
    assert isinstance(err, IndexError)

    with pytest.raises(ValueError):
        raise ArraySizeError(0)

    with pytest.raises(IndexError):
        raise ArraySizeError(0)
