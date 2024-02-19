import metran


def test_metran_solve_scipy(mt_init):
    mt_init.solve()


def test_metran_solve_lmfit(mt_init):
    mt_init.solve(solver=metran.solver.LmfitSolve)


def test_metran_state_means(mt):
    _ = mt.get_state_means()


def test_metran_simulated_means(mt):
    _ = mt.get_simulated_means()


def test_metran_get_simulation(mt):
    _ = mt.get_simulation("B21B0214005")


def test_metran_decompose_simulation(mt):
    _ = mt.decompose_simulation("B21B0214001")


def test_metran_get_state(mt):
    _ = mt.get_state(0)


def test_metran_masked_oseries(mt):
    proj1 = mt.get_simulation("B21B0214005")
    oseries = mt.get_observations()
    mask = (0 * oseries).astype(bool)
    mask.loc["1997-8-28", "B21B0214005"] = True
    mt.mask_observations(mask)
    proj2 = mt.get_simulation("B21B0214005")
    mt.unmask_observations()
    assert (proj1 != proj2).any().any()
