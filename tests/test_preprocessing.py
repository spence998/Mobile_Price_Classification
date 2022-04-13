import numpy as np
import numpy.testing
import pandas as pd
import pytest

from data_preprocessing import preprocessing


# =================================== UNIT TESTS LIST===========================================
# ===== read_csv:                        1
# ===== list_numerical_values:           2,3
# ===== list_non_numerical_values:       4,5
# ===== fill_missing_with_value:         6,7,8,9
# ===== fill_missing_with_mean:          10
# ===== one_hot_encoder                  11,12
# ===== ordinal_encoder                  13,14
# ===== remove_single_valued_columns     #todo
# ==============================================================================================

@pytest.mark.parametrize("csv_file_name, expected_df",
                         [
                             (
                                     "basic_df.csv",
                                     pd.DataFrame(
                                         data=[
                                             [1, 2, 3, 4],
                                             [5, 6, 7, 8],
                                             [9, 10, 11, 12]
                                         ],
                                         columns=["a", "b", "c", "d"]
                                     )
                             ),
                             (
                                     "basic_df_missings.csv",
                                     pd.DataFrame(
                                         data=[
                                             [1.0, 2.0, 3, 4.0],
                                             [5.0, np.nan, 7, 8.0],
                                             [9.0, 10.0, 11, 12.0],
                                             [np.nan, 14.0, 15, np.nan],
                                         ],
                                         columns=["a", "b", "c", "d"]
                                     )
                             )

                         ], ids=["Reading basic dataframe",
                                 "Reading dataframe with missings"])
def test_read_csv(csv_file_name, expected_df):
    """" 1) Tests for read_csv_file"""
    df = preprocessing.read_csv_file("test_data", csv_file_name)
    pd.testing.assert_frame_equal(df, df)


@pytest.mark.parametrize("df",
                         [(pd.DataFrame(
                             data=[
                                 [1, 2, 3, 4, "a", "b"],
                                 [5, np.nan, 7, 8, "c", "d"],
                                 [9, 10, 11, 12, "e", np.nan],
                                 [np.nan, 14, 15, np.nan, np.nan, "F"]
                             ],
                             columns=["a", "b", "c", "d", "e", "f"]))
                         ],
                         ids=["Testing return of columns"])
def test_list_numerical_values_1(df):
    """2) Tests for list_of_numerical_values"""
    _, df_numerical_columns = preprocessing.list_numerical_values(df)
    assert False not in (df_numerical_columns == ["a", "b", "c", "d"])


@pytest.mark.parametrize("df",
                         [(pd.DataFrame(
                             data=[
                                 [1, 2, 3, 4, "a", "b"],
                                 [5, np.nan, 7, 8, "c", "d"],
                                 [9, 10, 11, 12, "e", np.nan],
                                 [np.nan, 14, 15, np.nan, np.nan, "F"]
                             ],
                             columns=["a", "b", "c", "d", "e", "f"]))
                         ],
                         ids=["Testing return of dataframe"])
def test_list_numerical_values_2(df):
    """3) Tests for list_of_numerical_values"""
    df_numerical, _ = preprocessing.list_numerical_values(df)
    df.drop(["e", "f"], axis=1, inplace=True)
    pd.testing.assert_frame_equal(df_numerical, df)


@pytest.mark.parametrize("df",
                         [(pd.DataFrame(
                             data=[
                                 [1, 2, 3, 4, "a", "b"],
                                 [5, np.nan, 7, 8, "c", "d"],
                                 [9, 10, 11, 12, "e", np.nan],
                                 [np.nan, 14, 15, np.nan, np.nan, "F"]
                             ],
                             columns=["a", "b", "c", "d", "e", "f"]))
                         ],
                         ids=["Testing return of columns"])
def test_list_nonnumerical_values_1(df):
    """4) Test for list_of_non_numerical_values"""
    _, df_nonnumerical_columns = preprocessing.list_non_numerical_values(df)
    assert False not in (df_nonnumerical_columns == ["e", "f"])


@pytest.mark.parametrize("df",
                         [(pd.DataFrame(
                             data=[
                                 [1, 2, 3, 4, "a", "b"],
                                 [5, np.nan, 7, 8, "c", "d"],
                                 [9, 10, 11, 12, "e", np.nan],
                                 [np.nan, 14, 15, np.nan, np.nan, "F"]
                             ],
                             columns=["a", "b", "c", "d", "e", "f"]))
                         ],
                         ids=["Testing return of dataframe"])
def test_list_nonnumerical_values_2(df):
    """5) Test for list_of_non_numerical_values"""
    df_numerical, _ = preprocessing.list_non_numerical_values(
        df)
    df.drop(["a", "b", "c", "d"], axis=1, inplace=True)
    pd.testing.assert_frame_equal(df_numerical, df)


@pytest.mark.parametrize("df",
                         [(pd.DataFrame(
                             data=[
                                 [1, 2, 3, 4, "a", "b"],
                                 [5, np.nan, 7, 8, "c", "d"],
                                 [9, 10, 11, 12, "e", np.nan],
                                 [np.nan, 14, 15, np.nan, np.nan, "F"]
                             ],
                             columns=["a", "b", "c", "d", "e", "f"]))
                         ],
                         ids=["with an int (9)"])
def test_fill_missings_1(df):
    """6) Test for fill_missings_with_value"""
    pd.testing.assert_frame_equal(
        preprocessing.fill_missing_with_value(df, 9),
        pd.DataFrame(
            data=[
                [1, 2, 3, 4, "a", "b"],
                [5, 9, 7, 8, "c", "d"],
                [9, 10, 11, 12, "e", 9],
                [9, 14, 15, 9, 9, "F"]
            ],
            columns=["a", "b", "c", "d", "e", "f"]
        ), check_dtype=False
    )


@pytest.mark.parametrize("df",
                         [(pd.DataFrame(
                             data=[
                                 [1, 2, 3, 4, "a", "b"],
                                 [5, np.nan, 7, 8, "c", "d"],
                                 [9, 10, 11, 12, "e", np.nan],
                                 [np.nan, 14, 15, np.nan, np.nan, "F"]
                             ],
                             columns=["a", "b", "c", "d", "e", "f"]))
                         ],
                         ids=["with a float (16.7)"])
def test_fill_missings_2(df):
    """7) Test for fill_missing_with_value"""
    pd.testing.assert_frame_equal(
        preprocessing.fill_missing_with_value(df, 16.7),
        pd.DataFrame(
            data=[
                [1, 2, 3, 4, "a", "b"],
                [5, 16.7, 7, 8, "c", "d"],
                [9, 10, 11, 12, "e", 16.7],
                [16.7, 14, 15, 16.7, 16.7, "F"]
            ],
            columns=["a", "b", "c", "d", "e", "f"]
        ), check_dtype=False
    )


@pytest.mark.parametrize("df",
                         [(pd.DataFrame(
                             data=[
                                 [1, 2, 3, 4, "a", "b"],
                                 [5, np.nan, 7, 8, "c", "d"],
                                 [9, 10, 11, 12, "e", np.nan],
                                 [np.nan, 14, 15, np.nan, np.nan, "F"]
                             ],
                             columns=["a", "b", "c", "d", "e", "f"]))
                         ],
                         ids=["with a string 'test_word'"])
def test_fill_missings_3(df):
    """8) Test for fill_missing_with_value"""
    pd.testing.assert_frame_equal(
        preprocessing.fill_missing_with_value(df, 'test_word'),
        pd.DataFrame(
            data=[
                [1, 2, 3, 4, "a", "b"],
                [5, "test_word", 7, 8, "c", "d"],
                [9, 10, 11, 12, "e", "test_word"],
                ["test_word", 14, 15, "test_word", "test_word", "F"]
            ],
            columns=["a", "b", "c", "d", "e", "f"]
        ), check_dtype=False
    )


@pytest.mark.parametrize("df",
                         [(pd.DataFrame(
                             data=[
                                 [1, 2, 3, 4, "a", "b"],
                                 [5, np.nan, 7, 8, "c", "d"],
                                 [9, 10, 11, 12, "e", np.nan],
                                 [np.nan, 14, 15, np.nan, np.nan, "F"]
                             ],
                             columns=["a", "b", "c", "d", "e", "f"]))
                         ],
                         ids=["using a dictionary"])
def test_fill_missings_4(df):
    """9) Test for fill_missing_with_value"""
    pd.testing.assert_frame_equal(
        preprocessing.fill_missing_with_value(
            df,
            {
                "a": 99,
                "b": "cc",
                "e": "aa",
                "f": 16.7
            }
        ),
        pd.DataFrame(
            data=[
                [1, 2, 3, 4, "a", "b"],
                [5, "cc", 7, 8, "c", "d"],
                [9, 10, 11, 12, "e", 16.7],
                [99, 14, 15, np.nan, "aa", "F"]
            ],
            columns=["a", "b", "c", "d", "e", "f"]
        ), check_dtype=False
    )


@pytest.mark.parametrize("df",
                         [(pd.DataFrame(
                             data=[
                                 [1, 2, 3, 4, "a", "b"],
                                 [5, np.nan, 7, 8, "c", "d"],
                                 [9, 10, 11, 12, "e", np.nan],
                                 [np.nan, 14, 15, np.nan, np.nan, "F"]
                             ],
                             columns=["a", "b", "c", "d", "e", "f"]))
                         ])
def test_fill_missings_with_value_mean(df):
    """10) Test for fill_missing_with_value"""
    pd.testing.assert_frame_equal(
        preprocessing.fill_missing_with_mean(df),
        pd.DataFrame(
            data=[
                [1, 2, 3, 4, "a", "b"],
                [5, 8.666667, 7, 8, "c", "d"],
                [9, 10, 11, 12, "e", np.nan],
                [5.0, 14, 15, 8, np.nan, "F"]
            ],
            columns=["a", "b", "c", "d", "e", "f"]
        ), check_dtype=False
    )


@pytest.mark.parametrize("df, expected_df",
                         [
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a"],
                                             ["c"],
                                             [""],
                                             ["e"],
                                         ],
                                         columns=["a"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [0.0, 1.0, 0.0, 0.0],
                                             [0.0, 0.0, 1.0, 0.0],
                                             [1.0, 0.0, 0.0, .0],
                                             [0.0, 0.0, .0, 1.0]
                                         ],
                                         columns=["a_", "a_a", "a_c", "a_e"]
                                     )
                             ),
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a", "b"],
                                             ["c", "d"],
                                             ["e", ""],
                                             ["", "f"]
                                         ],
                                         columns=["a", "b"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                                         ],
                                         columns=["a_", "a_a", "a_c", "a_e",
                                                  "b_", "b_b", "b_d", "b_f"]
                                     )
                             ),
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a", 1, 4.0],
                                             ["c", 2, 6.0],
                                             ["e", 3, 7.0],
                                             ["", 4, 9.0]
                                         ],
                                         columns=["a", "b", "c"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [1, 4.0, 0.0, 1.0, 0.0, 0.0],
                                             [2, 6.0, 0.0, 0.0, 1.0, 0.0],
                                             [3, 7.0, 0.0, 0.0, 0.0, 1.0],
                                             [4, 9.0, 1.0, 0.0, 0.0, 0.0]
                                         ],
                                         columns=["b", "c", "a_", "a_a", "a_c", "a_e"]
                                     )
                             ),
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a", 1, 4.0],
                                             ["c", 2, 6.0],
                                             ["a", 3, 7.0],
                                             ["", 4, 9.0]
                                         ],
                                         columns=["a", "b", "c"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [1, 4.0, 0.0, 1.0, 0.0],
                                             [2, 6.0, 0.0, 0.0, 1.0],
                                             [3, 7.0, 0.0, 1.0, 0.0],
                                             [4, 9.0, 1.0, 0.0, 0.0]
                                         ],
                                         columns=["b", "c", "a_", "a_a", "a_c"]
                                     )
                             ),

                         ],
                         ids=["One categorical column",
                              "Two categorical columns",
                              "Additional int & float column",
                              "Multi type columns & repeated values"])
def test_one_hot_encoding_1(df, expected_df):
    output_df, _ = preprocessing.one_hot_encoder(df)
    pd.testing.assert_frame_equal(output_df, expected_df)


@pytest.mark.parametrize("df, expected_df",
                         [
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a"],
                                             ["b"],
                                             ["c"],
                                             [""],
                                         ],
                                         columns=["a"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [0.0, 1.0, 0.0, 0.0],
                                             [0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 1.0],
                                             [1.0, 0.0, 0.0, 0.0]
                                         ],
                                         columns=["a_", "a_a", "a_b", "a_c"]
                                     )
                             ),
                         ],
                         ids=["Encoding testing"])
def test_one_hot_encoding_2(df, expected_df):
    df_copy = df.copy()
    _, enc = preprocessing.one_hot_encoder(df)
    output_df = enc.transform(df_copy)
    np.testing.assert_array_equal(expected_df, output_df)


@pytest.mark.parametrize("df, expected_df",
                         [
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a"],
                                             ["c"],
                                             [""],
                                             ["e"],
                                         ],
                                         columns=["a"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [1.0],
                                             [2.0],
                                             [0.0],
                                             [3.0]
                                         ],
                                         columns=["a"]
                                     )
                             ),
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a", "b"],
                                             ["c", "d"],
                                             ["e", ""],
                                             ["", "f"]
                                         ],
                                         columns=["a", "b"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [1.0, 1.0],
                                             [2.0, 2.0],
                                             [3.0, 0.0],
                                             [0.0, 3.0]
                                         ],
                                         columns=["a", "b"]
                                     )
                             ),
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a", 1, 4.0],
                                             ["c", 2, 6.0],
                                             ["e", 3, 7.0],
                                             ["", 4, 9.0]
                                         ],
                                         columns=["a", "b", "c"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [1, 4.0, 1.0],
                                             [2, 6.0, 2.0],
                                             [3, 7.0, 3.0],
                                             [4, 9.0, 0.0]
                                         ],
                                         columns=["b", "c", "a"]
                                     )
                             ),
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a", 1, 4.0],
                                             ["c", 2, 6.0],
                                             ["a", 3, 7.0],
                                             ["", 4, 9.0]
                                         ],
                                         columns=["a", "b", "c"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [1, 4.0, 1.0],
                                             [2, 6.0, 2.0],
                                             [3, 7.0, 1.0],
                                             [4, 9.0, 0.0]
                                         ],
                                         columns=["b", "c", "a"]
                                     )
                             ),

                         ],
                         ids=["One categorical column",
                              "Two categorical columns",
                              "Additional int & float column",
                              "Multi type columns & repeated values"])
def test_ordinal_encoding_1(df, expected_df):
    output_df, _ = preprocessing.ordinal_encoder(df)
    pd.testing.assert_frame_equal(output_df, expected_df)


@pytest.mark.parametrize("df, expected_df",
                         [
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a"],
                                             ["b"],
                                             ["c"],
                                             [""],
                                         ],
                                         columns=["a"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [1.0],
                                             [2.0],
                                             [3.0],
                                             [0.0]
                                         ],
                                         columns=["a"]
                                     )
                             ),
                             (
                                     pd.DataFrame(
                                         data=[
                                             ["a", 1],
                                             ["b", 2],
                                             ["c", 3],
                                             ["", 4],
                                         ],
                                         columns=["a", "b"]
                                     ),
                                     pd.DataFrame(
                                         data=[
                                             [1.0, 1.0],
                                             [2.0, 2.0],
                                             [3.0, 3.0],
                                             [0.0, 4.0]
                                         ],
                                         columns=["a", "b"]
                                     )
                             )
                         ],
                         ids=["Encoding testing", "Encoding multiple columns"])
def test_ordinal_encoding_2(df, expected_df):
    df_copy = df.copy()
    _, enc = preprocessing.ordinal_encoder(df)
    output_df = enc.transform(df_copy)
    print(expected_df.values)
    print(output_df)
    print("--------------------------")
    np.testing.assert_array_equal(expected_df.values, output_df)
