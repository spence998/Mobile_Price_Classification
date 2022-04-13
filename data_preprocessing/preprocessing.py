import os
from typing import Union, List, Dict

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# ================================= PRE-PROCESSING FUNCTIONS====================================
# TODO: PUT LOGGING IN ALL FUNCTIONS
# ===== Read from csv
# ===== save_as_parquet #TODO NEED TO TEST
# ===== list_numerical_values
# ===== list_non_numerical_values
# ===== fill_missing_with_value
# ===== fill_missing_with_mean
# ===== one_hot_encode #TODO Figure out how to get the encoder to work in a pipeline
# ===== ordinal_encoder #TODO Figure out how to get the encoder to work in a pipeline
# ===== remove_single_valued_columns
# TODO: COMMENT/TEST
# ===== create_validation_dset
# ===== _train_test_split
# TODO: CREATE
# https://scikit-learn.org/stable/modules/preprocessing.html
# ===== K-bins discretization
# ===== Normalisation / Standardisation
# https://scikit-learn.org/stable/modules/impute.html
# ===== Imputation of missing values
# https://scikit-learn.org/stable/modules/unsupervised_reduction.html
# ===== Dimensionality reduction
# https://scikit-learn.org/stable/modules/preprocessing_targets.html
# ===== Transforming a predicted target
# ===== Any other preprocessing and manipulation that needs to be done before the data is used
# ==============================================================================================


def read_csv_file(folder_path: str, file_name: str) -> pd.DataFrame:
    """Reads a csv file into a pandas dataframe"""
    return pd.read_csv(os.path.join(folder_path, file_name))


def save_as_parquet(df: pd.DataFrame, folder_path: str, file_name: str):
    """Saves a pandas dataframe as a parquet which can be read much faster than a csv"""
    df.to_parquet(os.path.join(folder_path, file_name), index=False)


def list_numerical_values(df: pd.DataFrame) -> (pd.DataFrame, List[str]):
    """Returns the numerical columns of a dataframe and
    a list of the numerical column names
    """
    df_numerical = df.select_dtypes(include=[int, float])
    df_numerical_columns = df_numerical.columns.values
    return df_numerical, df_numerical_columns


def list_non_numerical_values(df: pd.DataFrame) -> (pd.DataFrame, List[str]):
    """Returns the non-numerical columns of a dataframe and a list of the non-numerical
    column names
    """
    df_nonnumerical = df.select_dtypes(include=[object])
    df_nonnumerical_columns = df_nonnumerical.columns.values
    return df_nonnumerical, df_nonnumerical_columns


def fill_missing_with_value(df: pd.DataFrame, value: Union[Dict, int, float, str]) -> pd.DataFrame:
    """Fills missing values of a dataframe with either a chosen integer of a float value.

    :param df:    Input dataframe with missing values
    :param value: Can either be an individual value to fill missing values with or a dictionary of
                  {column names: values to replace for the column}
    :return:      Dataframe with missing values filled
    """
    return df.fillna(value)


def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the missing values of a dataframe with the average of each numeric column"""
    return df.fillna(df.mean())


def one_hot_encoder(df: pd.DataFrame,
                   columns: List[str] = None) -> (pd.DataFrame,
                                                  preprocessing.OneHotEncoder):
    """Replaces non-numerical variables in a pandas dataframe with one hot encoded columns

    :param df:      Dataframe with non-numerical columns to encode
    :param columns: List of columns to encode. Defaults to all non-numerical columns.
    :return:        tuple(Pandas dataframe containing the encoded features,
                          encoder)
    """
    if columns is None:
        df_non_numerical, columns = list_non_numerical_values(df)
    else:
        df_non_numerical = df[columns]
    enc = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_array = enc.fit_transform(df_non_numerical)
    encoded_df = pd.DataFrame(data=encoded_array, columns=enc.get_feature_names_out())
    df.drop(columns=columns, inplace=True)
    return df.join(encoded_df), enc


def ordinal_encoder(df: pd.DataFrame,
                    columns: List[str] = None,
                    categories: List[str] = 'auto',
                    unknown_value=-1) -> (pd.DataFrame,
                                          preprocessing.OrdinalEncoder):
    """Replaces non-numerical variables in a pandas dataframe with ordinal encoded columns
    When tranforming data, the encoder will convert all unknown_values to what the
    "unknown_value" variable is set to.

    :param df:              Dataframe containing non-numerical columns to encode
    :param columns:         List of columns to encode. Defaults to all non-numerical columns.
    :param categories:      #TODO Complete this and figure out what it does
    :param unknown_value:   When using the returned encoder for tranforming data, the encoder will
                            convert all unknown_values to what the "unknown_value" variable is
                            set to. Defaults to -1.
    :return:                tuple(Pandas dataframe containing the encoded features,
                                  encoder)
    """
    if columns is None:
        df_non_numerical, columns = list_non_numerical_values(df)
    else:
        df_non_numerical = df[columns]
    enc = preprocessing.OrdinalEncoder(
        categories=categories,
        handle_unknown='use_encoded_value',
        unknown_value=unknown_value,
    )
    encoded_array = enc.fit_transform(df_non_numerical)
    encoded_df = pd.DataFrame(data=encoded_array, columns=columns)
    df.drop(columns=columns, inplace=True)
    return df.join(encoded_df), enc


def remove_single_valued_columns(df: pd.DataFrame) -> (pd.DataFrame, List[str]):
    """Removes all columns in a dataframe that only contain one value (including missing value)

    :param df:  Dataframe to remove single valued columns
    :return:    tuple(Dataframe with no single valued columns,
                      list of removed columns)
    """
    removed_columns = []
    for column in df.columns:
        if len(set(df[column])) == 1 or (
                df[column].dtype == np.number and np.isnan(np.array(df[column].values)).all()
        ):
            df.drop(column, axis=1, inplace=True)
            removed_columns.append(column)
    return df, removed_columns


def create_validation_dset(df: pd.DataFrame, validation_size: Union[int, float] = 0.3):
    """Split dataset into random train and validation subsets.

    :param df:          Dataset to split
    :param test_size:   Proportion of the dataset to put in the validation dataset
    :return:            tuple(training dataset,
                              validation dataset)
    """
    return train_test_split(df, test_size=validation_size)


def _train_test_split(X: pd.DataFrame, y: pd.DataFrame, test_size: float, seed: int):
    """Splits the dataframe into the training and testing subsets.

    :param X:           Dataframe containing all columns except the target variable
    :param y:           Dataframe containing the target variable column
    :param test_size:   Proportion of the dataset to put in the test dataset
    :param seed:        Random seed for shuffling the data
    :return:            Split dataset into a test and train sample in the format:
                        tuple(X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def kbins_discretization(df):
    est = preprocessing.KBinsDiscretizer(n_bins=[3, 2], encode='ordinal', strategy='uniform')
    binned_df = est.fit_transform(df)
    print(pd.DataFrame(data=binned_df, columns=['a', 'b']))


df = pd.DataFrame(
    data=[
        ["1", "2", "1", "9"],
        ["5", "6", "7", "6"],
        ["9", "10", "11", "5"]
    ],
    columns=["a", "b", "c", "d"]
)
kbins_discretization(df[['a','b']].values)

# encoded_df, enc = ordinal_encoder(df)
# print(encoded_df.values.flatten())
# print(enc.transform(encoded_df.values.flatten()))

