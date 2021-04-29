from metran.factoranalysis import FactorAnalysis
import metran
import pytest
import numpy as np

from conftest import get_data

def test_fa_init():
    corr = np.array([[1, 0.8], [0.8, 1]])
    fa = FactorAnalysis()
    return fa, corr

def test_fa_eigval():
    fa, corr = test_fa_init()
    eigval, eigvec = fa._get_eigval(corr)
    assert np.allclose(eigval, np.array([1.8, 0.2]))
    return eigval, eigvec

def test_fa_maptest():
    fa, corr = test_fa_init()
    eigval, eigvec = test_fa_eigval()
    (nfactors, _) = fa._maptest(corr, eigvec, eigval)
    assert nfactors == 1

def test_fa_solve():
    series = get_data()
    mt = metran.Metran(series, name="B21B0214")
    fa = FactorAnalysis()
    factors = fa.solve(mt.oseries)
    assert factors.shape == (5, 1)