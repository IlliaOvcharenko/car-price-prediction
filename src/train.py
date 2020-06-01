import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer 
from sklearn.impute import MissingIndicator

from src.utils import zip_dataframes
from src.utils import unzip_dataframes


def preprocessing(train_df, test_df, funcs):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for func in funcs:
        train_df, test_df = func(train_df, test_df)
    return train_df, test_df

def cross_validate(
    model,
    train_df,
    kfold,
    metric,
    preproc_funcs,
    target="price",
    test_df=None,
    log_target=False,
    *args,
    **kwargs
):
    val_scores = []
    test_preds = []
    
    if isinstance(kfold, GroupKFold):
        splits = kfold.split(train_df, groups=kwargs["groups"])
    elif isinstance(kfold, StratifiedKFold):
        target_values = train_df[[target]]
        est = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile')
        stratify_on = est.fit_transform(target_values).T[0]
        splits = kfold.split(train_df, stratify_on)
    else:
        splits = kfold.split(train_df)

    for idx, (tr_idx, val_idx) in enumerate(splits):
        tr_df = train_df.iloc[tr_idx]
        val_df = train_df.iloc[val_idx]
        
        if test_df is not None:
            tr_df, zip_df = preprocessing(tr_df, zip_dataframes(val_df, test_df), preproc_funcs)
            val_df, ts_df = unzip_dataframes(zip_df)
        else:
            tr_df, val_df = preprocessing(tr_df, val_df, preproc_funcs)
        
        x_tr = tr_df.drop(columns=target).values
        y_tr = tr_df[target].values
        x_val = val_df.drop(columns=target).values
        y_val = val_df[target].values
        
        if log_target:
            y_tr = np.log(y_tr)
            y_val = np.log(y_val)
        
        model.fit(x_tr, y_tr)
        preds = model.predict(x_val)
        
        preds = np.exp(preds) if log_target else preds
        y_val = np.exp(y_val) if log_target else y_val
        
        fold_score = metric(y_val, preds)
        val_scores.append(fold_score)
        
        print(f"fold {idx+1} score: {fold_score}")

        if test_df is not None:
            x_ts = ts_df.drop(columns=target).values
            test_fold_preds = model.predict(x_ts)
            test_fold_preds = np.exp(test_fold_preds) if log_target else test_fold_preds
            test_preds.append(test_fold_preds)
            
    print(f"mean score: {np.mean(val_scores)}")
    print(f"score variance: {np.var(val_scores)}")

    if test_df is not None:
        return val_scores, test_preds
    
    return val_scores

def full_tain_pred(
    model,
    train_df,
    test_df,
    preproc_funcs,
    target="price",
):
    tr_df, ts_df = preprocessing(train_df, test_df, preproc_funcs)
    x_tr = tr_df.drop(columns=target).values
    y_tr = tr_df[target].values
    x_ts = ts_df.values
    
    model.fit(x_tr, y_tr)
    preds = model.predict(x_ts)
    return preds
