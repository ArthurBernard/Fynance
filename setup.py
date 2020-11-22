#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2019-02-19 19:54:59
# @Last modified by: ArthurBernard
# @Last modified time: 2020-11-22 10:58:32

""" Setup script. """

# Built-in packages
import sys
from setuptools import setup, find_packages
from distutils.extension import Extension
from distutils.command.build_ext import build_ext

# Third party packages
import numpy

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
USE_CYTHON = 'auto'

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Financial and Insurance Industry',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python',
    'Programming Language :: Cython',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Office/Business :: Financial',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

MAJOR = 1
MINOR = 1
PATCH = 0
ISRELEASED = False
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, PATCH)


def get_version_info():
    FULLVERSION = VERSION
    GIT_REVISION = ""

    if not ISRELEASED:
        FULLVERSION += '.dev' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='fynance/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM FYNANCE SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


write_version_py()

DESCRIPTION = 'Python and Cython scripts of machine learning, econometrics '
DESCRIPTION += 'and statistical features for financial analysis [In progress]'

build_requires = [
    'Cython>=0.29.0',
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
        USE_CYTHON = True

    except ImportError:
        if not USE_CYTHON == 'auto':
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

if 'build_ext' in sys.argv[1:] or USE_CYTHON or USE_CYTHON == 'auto':
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
        'fynance.features.metrics_cy',
        ['fynance/features/metrics_cy' + ext],
        include_dirs=[numpy.get_include(), '.']
    ),
    Extension(
        'fynance.features.momentums_cy',
        ['fynance/features/momentums_cy' + ext],
        include_dirs=[numpy.get_include(), '.']
    ),
    Extension(
        'fynance.estimator.estimator_cy',
        ['fynance/estimator/estimator_cy' + ext],
        include_dirs=[numpy.get_include(), '.']
    ),
    Extension(
        'fynance.features.roll_functions_cy',
        ['fynance/features/roll_functions_cy' + ext],
        include_dirs=[numpy.get_include(), '.']
    ),
    Extension(
        'fynance.algorithms.browsers_cy',
        ['fynance/algorithms/browsers_cy' + ext],
        include_dirs=[numpy.get_include(), '.']
    ),
]

if USE_CYTHON or USE_CYTHON == 'auto':
    ext_modules = cythonize(extensions, annotate=True)

else:
    ext_modules = extensions

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='fynance',
    version=VERSION,
    description=DESCRIPTION,
    long_description=str(long_description),
    license='MIT',
    url='https://github.com/ArthurBernard/Fynance',
    download_url='https://pypi.org/project/fynance/',
    project_urls={
        'Documentation': 'https://fynance.readthedocs.io/',
        'Source Code': 'https://github.com/ArthurBernard/Fynance/'
    },
    author=['Arthur Bernard'],
    author_email='arthur.bernard.92@gmail.com',
    packages=find_packages(),  # ['fynance'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=build_requires,
    classifiers=CLASSIFIERS,
)
