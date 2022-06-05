from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate


def svm_on_unbalanced_df(df: pd.DataFrame, target_col: str) -> (Any, np.array, np.array):
    svc_model = LinearSVC(class_weight='balanced')
    cols = [c for c in df.columns if c != target_col]
    X = df[cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.3)
    svc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    return svc_model, y_pred, y_test


def svm_on_balanced_df(df: pd.DataFrame, target_col: str, scoring: Optional[List] = None, cv=10, verbose=True, max_iter=1000)-> (Dict, Dict) :
    """
    The function returns the evaluation scores after apply K-fold cross validation
    :param df:
    :param target_col:
    :param scoring:
    :param cv:
    :param verbose:
    :return:
    """
    if not scoring:
        scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'roc_auc', 'matthews_corrcoef']

    svc_model = LinearSVC(max_iter=max_iter)
    cols = [c for c in df.columns if c != target_col]
    X = df[cols]
    y = df[target_col]

    scores = cross_validate(svc_model, X, y, scoring=scoring, cv=cv, n_jobs=-1, verbose=verbose)
    mean_scores = {name: np.mean(arr) for name, arr in scores.items()}
    return scores, mean_scores