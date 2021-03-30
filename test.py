from metran import Metran
import pandas as pd

series = []
nts = 5
for s in range(nts):
    ts = pd.read_csv("examples/data/B21B021400" + str(s+1) + "_res.csv",
                     header=0, index_col=0,
                     parse_dates=True, infer_datetime_format=True,
                     dayfirst=True)
    series.append(ts)

mt = Metran(series)
mt.solve()
