from pathlib import Path
from typing import Any, List

import pytest
from numpy import array
from pandas import Series, read_csv

import metran


@pytest.fixture
def series_list() -> List[Series]:
    path = Path(__file__).parent.parent / "examples/data"
    seriesl = [
        read_csv(
            fi,
            header=0,
            index_col=0,
            parse_dates=True,
            date_format="%Y-%m-%d",
            names=[fi.stem.split("_")[0]],
        ).squeeze()
        for fi in path.glob("*_res.csv")
    ]
    return seriesl


@pytest.fixture
def mt_init(series_list) -> metran.Metran:
    """Fixture that yields initialized metran object"""
    return metran.Metran(series_list, name="B21B0214")


@pytest.fixture
def mt(mt_init) -> metran.Metran:
    """Fixture that yields solved metran object"""
    mt_init.solve()
    return mt_init


@pytest.fixture
def corr() -> Any:
    return array([[1.0, 0.8], [0.8, 1.0]], dtype=float)
