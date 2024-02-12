from pathlib import Path
from typing import Any, List

import pytest
from numpy import array
from pandas import Series, read_csv

import metran

seriesl = [
    read_csv(
        fi,
        header=0,
        index_col=0,
        parse_dates=True,
        date_format="%Y-%m-%d",
        names=[fi.stem.split("_")[0]],
    )
    for fi in Path("../examples/data").glob("*_res.csv")
]


@pytest.fixture
def series_list() -> List[Series]:
    return seriesl


@pytest.fixture
def mt_init(seriesl) -> metran.Metran:
    """Fixture that yields initialized metran object"""
    return metran.Metran(seriesl, name="B21B0214")


@pytest.fixture
def mt(mt_init) -> metran.Metran:
    """Fixture that yields solved metran object"""
    return mt_init.solve()


@pytest.fixture
def corr() -> Any:
    return array([[1.0, 0.8], [0.8, 1.0]], dtype=float)
