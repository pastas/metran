import metran
import pytest

from conftest import get_data


def test_metran_init():
    series = get_data()
    mt = metran.Metran(series, name="B21B0214")
    return mt


def test_metran_solve_scipy():
    mt = test_metran_init()
    mt.solve()
    return


def test_metran_solve_lmfit():
    mt = test_metran_init()
    mt.solve(solver=metran.solver.LmfitSolve)
    return


def test_metran_state_means(mt):
    _ = mt.get_state_means()
    return


def test_metran_projected_means(mt):
    _ = mt.get_simulated_means()
    return


def test_metran_get_projection(mt):
    _ = mt.get_simulation("B21B0214005")
    return


def test_metran_decompose_projection(mt):
    _ = mt.decompose_simulation("B21B0214001")
    return


def test_metran_get_state(mt):
    _ = mt.get_state(0)
    return


def test_metran_masked_oseries(mt):
    proj1 = mt.get_simulation("B21B0214005")
    oseries = mt.get_observations()
    mask = (0 * oseries).astype(bool)
    mask.loc["1997-8-28", "B21B0214005"] = True
    mt.mask_observations(mask)
    proj2 = mt.get_simulation("B21B0214005")
    mt.unmask_observations()
    assert (proj1 != proj2).any().any()
    return
