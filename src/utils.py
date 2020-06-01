import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.cli import tqdm
from pathlib import Path
from functools import partial
from sklearn.metrics import mean_squared_error


all_rows = partial(pd.option_context, 'display.max_rows', None, 'display.max_columns', None)

def fprint(df):
    """ full print for pandas dataframes 
    """
    with all_rows():
        print(df)

def percent_of(scores, cl=1):
    return (scores == cl).sum() / len(scores)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def zip_dataframes(*dataframes):
    for idx, dataframe in enumerate(dataframes):
        dataframe["df_order"] = idx
    return pd.concat(dataframes)

def unzip_dataframes(dataframe):
    dataframes = []
    for n in dataframe["df_order"].unique().tolist():
        dataframes.append(dataframe[dataframe["df_order"] == n].drop(columns="df_order"))
    return dataframes
    

def create_submit_df(test_df, preds):
    submit_df = pd.DataFrame({
        "Id": test_df["index"],
        "Predicted": preds,
    })
    return submit_df
