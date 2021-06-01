[![metran](https://github.com/pastas/metran/actions/workflows/ci.yml/badge.svg)](https://github.com/pastas/metran/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/metran/badge/?version=latest)](https://metran.readthedocs.io/en/latest/?badge=latest)

# Metran

Metran is a package for performing multivariate timeseries analysis using a technique called dynamic factor modelling. It can be used to describe the variation among many variables in terms of a few underlying but unobserved variables called factors.

## Installation

To install Metran, a working version of Python 3.7 or 3.8 has to be installed on your computer. We recommend using the Anaconda Distribution with Python 3.7 as it includes most of the python package dependencies and the Jupyter Notebook software to run the notebooks. However, you are free to install any Python distribution you want.

To install `metran`, clone the repository and type the following command

`pip install .`

To install in development mode, use:

`pip install -e .`

## Dependencies

Metran has the following dependencies which are automatically installed if
not already available:

-   numpy>=1.16.5
-   pandas>=1.0
-   scipy>=1.1
-   matplotlib>=3.0
-   pastas>=0.16.0
-   numba

## Examples

For a brief introduction of the theory behind Metran on multivariate timeseries analysis with
dynamic factor modeling see the notebook:

-   [The Dynamic Factor Model](https://github.com/pastas/metran/blob/main/examples/dynamic_factor_model.ipynb)

A practical real-world example, as published in Stromingen (Van Geer, 2015), is given in the following notebook:

-   [Metran practical example](https://github.com/pastas/metran/blob/main/examples/metran_practical_example.ipynb)

A notebook on how to use [Pastas](https://github.com/pastas/pastas) models output with Metran:

-   [Pastas Metran example](https://github.com/pastas/metran/blob/main/examples/pastas_metran_example.ipynb)

## References

-   Berendrecht, W.L. (2004). State space modeling of groundwater fluctuations. 
-   Berendrecht, W.L., F.C. van Geer (2016). A dynamic factor modeling framework for analyzing multiple groundwater head series simultaneously, Journal of Hydrology, 536, pp. 50-60, doi:<http://dx.doi.org/10.1016/j.jhydrol.2016.02.028>.
-   Van Geer, F.C. en W.L. Berendrecht (2015) Meervoudige tijdreeksmodellen en de samenhang in stijghoogtereeksen. Stromingen 23 nummer 3, pp. 25-36.
