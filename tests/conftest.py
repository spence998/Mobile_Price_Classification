import pandas as pd
import numpy as np
import pytest


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame(
        data=[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ],
        columns=["a", "b", "c", "d"]
    )


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame(
        data=[
            [1, 2, 3, 4, "a", "b"],
            [5, np.nan, 7, 8, "c", "d"],
            [9, 10, 11, 12, "e", np.nan],
            [np.nan, 14, 15, np.nan, np.nan, "F"]
        ],
        columns=["a", "b", "c", "d", "e", "f"]
    )
