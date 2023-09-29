import os

import pandas as pd
import pytest

import metran


def get_data():
    datadir = "./examples/data"
    series = []
    files = [
        os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith("_res.csv")
    ]
    files.sort()
    for fi in files:
        name = fi.split(os.sep)[-1].split(".")[0].split("_")[0]
        ts = pd.read_csv(
            fi,
            header=0,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            dayfirst=True,
            names=[name],
        )
        series.append(ts)
    return series


@pytest.fixture(scope="module")
def mt(request):
    """Fixture that yields metran object"""
    series = get_data()
    mt = metran.Metran(series, name="B21B0214")
    mt.solve()
    yield mt
