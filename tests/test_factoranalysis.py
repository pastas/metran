import numpy as np

import metran
from metran.factoranalysis import FactorAnalysis


def test_fa_eigval(corr):
    fa = FactorAnalysis()
    eigval, _ = fa._get_eigval(corr)
    assert np.allclose(eigval, np.array([1.8, 0.2], dtype=float))


def test_fa_maptest(corr):
    fa = FactorAnalysis()
    eigval, eigvec = fa._get_eigval(corr)
    nfactors, _ = fa._maptest(corr, eigvec, eigval)
    assert nfactors == 1


def test_fa_solve(seriesl):
    mt = metran.Metran(seriesl, name="B21B0214")
    fa = FactorAnalysis()
    factors = fa.solve(mt.oseries)
    assert factors.shape == (5, 1)
