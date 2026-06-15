#!/usr/bin/env python3
"""Build Cython extensions.

Metadata lives in pyproject.toml. This file only handles the Cython
extension compilation with an auto-fallback to pre-compiled .c files.
"""

import sys
from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext

import numpy

USE_CYTHON = 'auto'

if USE_CYTHON == 'auto':
    try:
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext
        ext = '.pyx'
        USE_CYTHON = True
        print('Using Cython.')
    except ImportError:
        ext = '.c'
        USE_CYTHON = False
        print('Cython not found, using pre-compiled .c files.')
else:
    ext = '.c'

include_dirs = [numpy.get_include(), '.']

extensions = [
    # NOTE: the econometric (ARMA/GARCH) and estimator kernels were ported to
    # numba (@njit) in 2.1; only the features kernels remain in Cython.
    Extension(
        'fynance.features.metrics_cy',
        ['fynance/features/metrics_cy' + ext],
        include_dirs=include_dirs,
    ),
    Extension(
        'fynance.features.momentums_cy',
        ['fynance/features/momentums_cy' + ext],
        include_dirs=include_dirs,
    ),
]

if USE_CYTHON:
    ext_modules = cythonize(extensions, annotate=True)
else:
    ext_modules = extensions

cmdclass = {'build_ext': build_ext} if ('build_ext' in sys.argv[1:] or USE_CYTHON) else {}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
