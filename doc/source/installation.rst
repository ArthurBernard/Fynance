============
Installation
============

Requirements
------------

- Python **3.10** or later
- NumPy, SciPy, Numba, PyTorch (installed automatically via pip)

The package is **pure-Python** — there is no compile step. The numerical
kernels are Numba ``@njit`` functions, JIT-compiled on first call.

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

Verify the installation
-----------------------

.. code-block:: python

   import fynance
   print(fynance.__version__)
