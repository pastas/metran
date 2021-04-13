import pandas as pd
from metran import Metran

series = []
nts = 5
for s in range(nts):
    ts = pd.read_csv("examples/data/B21B021400" + str(s+1) + "_res.csv",
                     header=0, index_col=0, names=["B21B021400" + str(s+1)],
                     parse_dates=True, infer_datetime_format=True,
                     dayfirst=True)
    series.append(ts)
mt = Metran(series, name="B21B0214")
mt.solve()
# Get eigenvalues (can be used to plot scree plot, see e.g. Fig 2 in JoH paper)
eigval = mt.eigval
# Get all smoothed state means
states = mt.get_state_means()
# Get all (smoothed) projected state means
means = mt.get_projected_means()
# Get projected mean for specific series with/without confidence interval
proj = mt.get_projection("B21B0214001", ci=False)
# Decomposed projected mean for specific series
dec = mt.decompose_projection("B21B0214001")
# Get specific state mean with/without confidence interval
mt.get_state(0, ci=True)


