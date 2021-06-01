.. metran documentation master file, created by
   sphinx-quickstart on Fri Apr 16 17:08:09 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

Metran is a package for performing timeseries analysis on multiple
timeseries using dynamic factor models.

When modeling multiple groundwater time series within the same hydrological 
system, it often appears that these components show distinct correlations between 
locations. Usually large part of the correlation is caused by common input 
stresses like precipitation and evapotranspiration, which shows up within 
the deterministic components of the models.

The residual components of the univariate TFN models are often correlated as 
well. This means that there is spatial correlation which has not been captured 
by the deterministic component, e.g. because of errors in common input data or 
due to simplification of the hydrological model leading to misspecification of 
the deterministic component. We can exploit these correlations by modeling the 
series simultaneously with a dynamic factor model. Dynamic factor modeling 
(DFM) is a multivariate timeseries analysis technique used to describe the 
variation among many variables in terms of a few underlying but unobserved 
variables called factors.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   Getting Started <getting_started>
   Concepts <concepts>
   Examples <examples>
   API-docs <modules>

Indices and tables
==================

* :ref:`genindex`
