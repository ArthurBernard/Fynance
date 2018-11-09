# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
USE_CYTHON = True


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
        print('Use cython')
    except ImportError:
        if USE_CYTHON == 'auto':
            USE_CYTHON = False
            print("Can't use cython")
        else:
            print('raise')
            raise

cmdclass = { }

if USE_CYTHON:
    extensions = [
        Extension(
            "fynance.models.econometric_models", 
            ["fynance/models/econometric_models.pyx"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.tools.metrics", 
            ["fynance/tools/metrics.pyx"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.estimator.estimator_cy", 
            ["fynance/estimator/estimator_cy.pyx"],
            include_dirs=[numpy.get_include(), '.']
        ),
    ]
    
    cmdclass.update({ 'build_ext': build_ext })
else:
    extensions = [
        Extension(
            "fynance.models.econometric_models", 
            ["fynance/models/econometric_models.c"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.tools.metrics", 
            ["fynance/tools/metrics.c"],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            "fynance.estimator.estimator_cy", 
            ["fynance/estimator/estimator_cy.c"],
            include_dirs=[numpy.get_include(), '.']
        ),
    ]

setup(
    name='fynance',
    version='0.1',
    description='Some tools for time series applications, especialy in finance, \
        econometric, statistic and machine learning [In progress]',
    url='https://github.com/ArthurBernard/Fynance',
    author='Arthur Bernard',
    author_email='arthur.bernard.92@gmail.com',
    packages=['fynance'],
    cmdclass=cmdclass,
    ext_modules=cythonize(extensions, annotate=True),
)
