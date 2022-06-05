from typing import Union

import pandas as pd
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
import numpy as np


# def undersampling_nearmiss(X: Union[pd.DataFrame, np.array], y: Union[pd.DataFrame, np.array], version=3) -> (np.array, np.array):
#     nm = NearMiss(version=version, n_neighbors=3)
#     return nm.fit_resample(X, y)

def undersampling_nearmiss(df, target_col: str, version=3) -> pd.DataFrame:
    cols = [c for c in df.columns if c != target_col]
    nm = NearMiss(version=version, n_neighbors=3)
    x_nm, y_nm = nm.fit_resample(df[cols], df[target_col])
    near_miss_df = pd.DataFrame(np.c_[x_nm, y_nm], columns=df.columns)

    return near_miss_df.astype(df.dtypes.to_dict())

def oversampling_smote(X: Union[pd.DataFrame, np.array], y: Union[pd.DataFrame, np.array]) -> (np.array, np.array):
    sm = SMOTE()
    return sm.fit_resample(X, y)


def hibryd_sampling_with_smote(df: pd.DataFrame, target_col: str, n_per_class=200000) -> pd.DataFrame:
    """
    Applies first random under sample of the majority class and then applies SMOTE on the minority class.
    Returns a dataset of size 2 * n_per_class
    :param df: input dataframe
    :param target_col:
    :param n_per_class: default value 200000 (20%)
    :return:
    """
    isFraud = df[target_col]
    negative_df = df[~isFraud].sample(n=n_per_class)
    positive_df = df[isFraud]
    undersample_df = pd.concat([positive_df, negative_df])
    cols = [c for c in df.columns if c != target_col]
    x_smote, y_smote = oversampling_smote(undersample_df[cols], undersample_df[target_col])
    smote_df = pd.DataFrame(np.c_[x_smote, y_smote], columns=df.columns)
    return smote_df.astype(df.dtypes.to_dict())
