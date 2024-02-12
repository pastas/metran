import pandas as pd

import metran

# read data
series = []
nts = 5
for s in range(nts):
    ts = pd.read_csv(
        "./data/B21B021400" + str(s + 1) + "_res.csv",
        header=0,
        index_col=0,
        names=["B21B021400" + str(s + 1)],
        parse_dates=True,
        date_format="%d-%m-%Y"
    )
    series.append(ts)

# create model
mt = metran.Metran(series, name="B21B0214")
mt.solve()

# Get eigenvalues (can be used to plot scree plot, see e.g. Fig 2 in JoH paper)
eigval = mt.eigval

# Get all smoothed state means
states = mt.get_state_means()

# Get all (smoothed) projected state means
means = mt.get_simulated_means()

# Get projected mean for specific series with/without 95% confidence interval
sim = mt.get_simulation("B21B0214005", alpha=0.05)

# Decomposed projected mean for specific series
dec = mt.decompose_simulation("B21B0214001")

# Get specific state mean with/without 95% confidence interval
mt.get_state(0, alpha=0.05)

# remove outlier from series B21B0214005 at 1997-8-28
# and re-run smoother to get estimate of observation
# (Fig 3 in Stromingen without deterministic component)
sim = mt.get_simulation("B21B0214005", alpha=0.05)
sim.loc["1997"].plot()
oseries = mt.get_observations()
mask = (0 * oseries).astype(bool)
mask.loc["1997-8-28", "B21B0214005"] = True
mt.mask_observations(mask)
sim = mt.get_simulation("B21B0214005", alpha=0.05)
sim.loc["1997"].plot()
# unmask observations to get original observations
mt.unmask_observations()
