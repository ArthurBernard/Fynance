============
Installation
============

Requirements
------------

- Python **3.8** or later
- NumPy, SciPy, PyTorch (installed automatically via pip)
- A C compiler and Cython are needed only when building from source
  (pre-compiled binaries are bundled in the PyPI wheel)

Install with pip
----------------

.. code-block:: bash

   pip install fynance

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/ArthurBernard/Fynance.git
   cd Fynance
   pip install -e ".[dev]"
   python setup.py build_ext --inplace

The ``build_ext`` step compiles the Cython extensions (``estimator``,
``features``). It must be re-run after editing any ``.pyx`` file.

Verify the installation
-----------------------

.. code-block:: python

   import fynance
   print(fynance.__version__)
