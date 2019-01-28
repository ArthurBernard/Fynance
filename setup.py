#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
USE_CYTHON = 'auto'


from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension
import numpy

if USE_CYTHON or USE_CYTHON == 'auto':
    try:
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext
        print('Installation using cython')
    except ImportError:
        if USE_CYTHON == 'auto':
            USE_CYTHON = False
            print("Installation not using cython")
        else:
            print("Install cython or set USE_CYTHON = 'auto'")
            raise

cmdclass = { }

if USE_CYTHON:
    extensions = [
        Extension(
            "fynance.models.econometric_models_cy", 
            ["fynance/models/econometric_models_cy.pyx"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.tools.metrics_cy", 
            ["fynance/tools/metrics_cy.pyx"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.tools.momentums_cy", 
            ["fynance/tools/momentums_cy.pyx"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.estimator.estimator_cy", 
            ["fynance/estimator/estimator_cy.pyx"],
            include_dirs=[numpy.get_include(), '.']
        ),
    ]
    ext_modules = cythonize(extensions, annotate=True)
    cmdclass.update({ 'build_ext': build_ext })
else:
    extensions = [
        Extension(
            "fynance.models.econometric_models_cy", 
            ["fynance/models/econometric_models_cy.c"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.tools.metrics_cy", 
            ["fynance/tools/metrics_cy.c"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.tools.momentums_cy", 
            ["fynance/tools/momentums_cy.c"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.estimator.estimator_cy", 
            ["fynance/estimator/estimator_cy.c"],
            include_dirs=[numpy.get_include(), '.']
        ),
    ]
    ext_modules = extensions

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='fynance',
    version='1.0.0',
    description='Python and Cython scripts of machine learning, econometrics \
        and statistical tools for financial analysis [In progress]',
    long_description=str(long_description),
    url='https://github.com/ArthurBernard/Fynance',
    author=['Arthur Bernard'],
    author_email='arthur.bernard.92@gmail.com',
    packages=find_packages(), #['fynance'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=[
        'matplotlib>=3.0.1',
        'numpy>=1.15.3',
        'pandas>=0.23.4',
        'Cython>=0.29',
        'scipy>=1.2.0',
        'seaborn>=0.9.0'
    ],
    classifiers=[
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
    ],
)
