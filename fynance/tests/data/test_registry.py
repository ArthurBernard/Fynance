#!/usr/bin/env python3
# coding: utf-8

""" Tests for the data-source registry / dispatcher. """

# Third-party packages
import pytest

# Local packages
from fynance.data import get_source, load


def test_get_source_known():
    from fynance.data import CSVSource, ParquetSource

    assert isinstance(get_source("csv"), CSVSource)
    assert isinstance(get_source("parquet"), ParquetSource)


def test_get_source_unknown():
    with pytest.raises(ValueError):
        get_source("nope")


def test_load_unknown_extension():
    with pytest.raises(ValueError):
        load("data.xyz")
