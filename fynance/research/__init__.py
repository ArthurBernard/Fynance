#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Research harness (:mod:`fynance.research`).

A **data-agnostic** layer for running strategy experiments and emitting portable
result artifacts. fynance is *the tool*: it never stores results itself — every
artifact is written to a caller-provided ``output_dir`` (a downstream private
research repo points that wherever it wants). Built and tested on the synthetic
generators in :mod:`fynance.research.synthetic`; real-data adapters live
downstream, never here.

.. currentmodule:: fynance.research

"""

# Local
from . import experiment, guards, report, runner, synthetic
from .experiment import *
from .guards import *
from .report import *
from .runner import *
from .synthetic import *

__all__: list[str] = []
__all__ += experiment.__all__
__all__ += synthetic.__all__
__all__ += runner.__all__
__all__ += report.__all__
__all__ += guards.__all__
