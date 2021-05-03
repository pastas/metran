Getting Started
===============

This page explains how to get started with Metran.

Installing Metran
-----------------

To install Metran, a working version of Python 3.7 or 3.8 has to be installed on 
your computer. We recommend using the Anaconda Distribution with Python 3.7 as 
it includes most of the python package dependencies and the Jupyter Notebook 
software to run the notebooks. However, you are free to install any 
Python distribution you want. 

To install metran, clone the repository and type the following command from 
the module root directory:

.. code:: bash

    pip install .

To install in development mode, use:

.. code:: bash

    pip install -e .


Basic usage
-----------

To use Metran, import the metran package:

.. code:: python

    import metran

Create Metran model by passing a list of timeseries (measured heads, or 
timeseries of the residuals of Pastas timeseries models).

.. code:: python
    
    list_of_series  # this is a list of series, i.e. [series1, series2, ...]

    mt = metran.Metran(list_of_series)

To solve the model and determine the specific an dynamic factors:

.. code:: python

    mt.solve()

Plotting a simulation for one of the the timeseries:

.. code:: python

    ax = mt.plots.simulation("series1")
