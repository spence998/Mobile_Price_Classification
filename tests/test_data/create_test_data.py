import pandas as pd
import numpy as np


def fixture_df():
    return pd.DataFrame(
        data=[
            [1, 2, 3, 4, ],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ],
        columns=["a", "b", "c", "d"]
    )


def fixture_df_missings():
    return pd.DataFrame(
        data=[
            [1, 2, 3, 4],
            [5, np.nan, 7, 8],
            [9, 10, 11, 12],
            [np.nan, 14, 15, np.nan]
        ],
        columns=["a", "b", "c", "d"]
    )


def create_csv(df: pd.DataFrame, file_name: str):
    df().to_csv(file_name)


if __name__ == "__main__":
    create_csv(fixture_df, "basic_df.csv")
    create_csv(fixture_df_missings, "basic_df_missings.csv")
