#!/usr/bin/env python3
# coding: utf-8

""" Strategy orchestration layer.

.. currentmodule:: fynance.strategy

:class:`Strategy` composes the pipeline maillons (features, model, signal, cost)
into one runnable object — optional, never required.

"""

# Local packages
from .strategy import Strategy

__all__ = ['Strategy']
