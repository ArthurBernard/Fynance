#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
USE_CYTHON = 'auto'

import sys
from setuptools import setup, find_packages
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import numpy

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Financial and Insurance Industry',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python',
    'Programming Language :: Cython',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Office/Business :: Financial',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

MAJOR = 1
MINOR = 0
PATCH = 4
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, PATCH)

DESCRIPTION = 'Python and Cython scripts of machine learning, econometrics ' 
DESCRIPTION += 'and statistical tools for financial analysis [In progress]'

build_requires = [
    'matplotlib>=3.0.1',
    'numpy>=1.15.3',
    'pandas>=0.23.4',
    'scipy>=1.2.0',
    'seaborn>=0.9.0',
]

if USE_CYTHON or USE_CYTHON == 'auto':
    try:
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext
        ext = '.pyx'
        print('Using cython.')
    except ImportError:
        if USE_CYTHON:
            print("If USE_CYTHON is set to True, Cython is required to", 
                "compile fynance. Please install Cython or don't set", 
                "USE_CYTHON to True.")
            raise ImportError
        else:
            print('Not using cython.')
            ext = '.c'
            USE_CYTHON = False
else:
    ext = '.c'

if 'build_ext' in sys.argv[1:] or USE_CYTHON:
    cmdclass = {'build_ext': build_ext}
else:
    cmdclass = {}

extensions = [
    Extension(
        'fynance.models.econometric_models_cy', 
        ['fynance/models/econometric_models_cy' + ext],
        include_dirs=[numpy.get_include(), '.']
    ),
    Extension(
        'fynance.tools.metrics_cy', 
        ['fynance/tools/metrics_cy' + ext],
        include_dirs=[numpy.get_include(), '.']
    ),
    Extension(
        'fynance.tools.momentums_cy', 
        ['fynance/tools/momentums_cy' + ext],
        include_dirs=[numpy.get_include(), '.']
    ),
    Extension(
        'fynance.estimator.estimator_cy', 
        ['fynance/estimator/estimator_cy' + ext],
        include_dirs=[numpy.get_include(), '.']
    ),
]

if USE_CYTHON:
    ext_modules = cythonize(extensions, annotate=True)
else:
    ext_modules = extensions

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='fynance',
    version=VERSION,
    description=DESCRIPTION,
    long_description=str(long_description),
    license='MIT',
    url='https://github.com/ArthurBernard/Fynance',
    download_url='https://pypi.org/project/fynance/',
    author=['Arthur Bernard'],
    author_email='arthur.bernard.92@gmail.com',
    packages=find_packages(), #['fynance'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=build_requires,
    classifiers=CLASSIFIERS,
)
