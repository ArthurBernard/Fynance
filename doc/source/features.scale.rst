*******
 Scale 
*******

This module contains some tools to scale data set, such as standardization (:func:`~fynance.features.scale.standardization`) or normalization (:func:`~fynance.features.scale.normalization`).

There is also a scale object (:func:`~fynance.features.scale.Scale`) that allows you to scale your data with standardization or normalization and keep in memory the parameters used to scale the data. Such that you can rescale an other data set with the same parameters or you can also revert the scale transformation.

.. currentmodule:: fynance.features.scale

Scale functions
===============

.. autosummary::
   :toctree: generated/

   fynance.features.scale.normalization
   fynance.features.scale.standardization

Scale object
============

.. autosummary::
   :toctree: generated/

   fynance.features.scale.Scale
