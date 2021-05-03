def test_scree_plot(mt):
    mt.plots.scree_plot()


def test_plot_state_means(mt):
    mt.plots.state_means(adjust_height=True)


def test_plot_simulation(mt):
    mt.plots.simulation("B21B0214003")


def test_plot_simulations(mt):
    mt.plots.simulations()


def test_plot_decomposition(mt):
    mt.plots.decomposition("B21B0214003", adjust_height=True)


def test_plot_decompositions(mt):
    mt.plots.decompositions()
