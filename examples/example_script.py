import pandas as pd
import metran

# read data
series = []
nts = 5
for s in range(nts):
    ts = pd.read_csv("examples/data/B21B021400" + str(s + 1) + "_res.csv",
                     header=0, index_col=0, names=["B21B021400" + str(s + 1)],
                     parse_dates=True, infer_datetime_format=True,
                     dayfirst=True)
    series.append(ts)

# create model
mt = metran.Metran(series, name="B21B0214")
mt.solve()

# Get eigenvalues (can be used to plot scree plot, see e.g. Fig 2 in JoH paper)
eigval = mt.eigval

# Get all smoothed state means
states = mt.get_state_means()

# Get all (smoothed) projected state means
means = mt.get_projected_means()

# Get projected mean for specific series with/without confidence interval
proj = mt.get_projection("B21B0214005", ci=True)

# Decomposed projected mean for specific series
dec = mt.decompose_projection("B21B0214001")

# Get specific state mean with/without confidence interval
mt.get_state(0, ci=True)

# remove outlier from series B21B0214005 at 1997-8-28
# and re-run smoother to get estimate of observation
# (Fig 3 in Stromingen without deterministic component)
proj = mt.get_projection("B21B0214005", ci=True)
proj.loc["1997"].plot()
oseries = mt.get_observations()
mask = (0 * oseries).astype(bool)
mask.loc["1997-8-28", "B21B0214005"] = True
mt.mask_observations(mask)
proj = mt.get_projection("B21B0214005", ci=True)
proj.loc["1997"].plot()
# unmask observations to get original observations
mt.unmask_observations()
