import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer 
from sklearn.impute import MissingIndicator


cat_features = ["type", "gearbox", "model", "fuel", "brand", "city"]
cont_missing_features = ["engine_capacity", "damage", "insurance_price", "latitude", "longitude"]
cat_missing_features = ["type", "gearbox", "model", "fuel", "city"]

def impute_nan_with_zero(train_df, test_df):
    for cat_feature in cat_features:
        train_df[cat_feature] = train_df[cat_feature].fillna("nan")
        test_df[cat_feature] = test_df[cat_feature].fillna("nan")
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    return train_df, test_df

def impute_nan(train_df, test_df):
    for cont_missing_feature in cont_missing_features:
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp.fit(pd.concat([train_df, test_df])[[cont_missing_feature]])
        train_df[cont_missing_feature] = imp.transform(train_df[[cont_missing_feature]])
        test_df[cont_missing_feature] = imp.transform(test_df[[cont_missing_feature]])

    for cat_missing_feature in cat_missing_features:
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="nan")

        imp.fit(pd.concat([train_df, test_df])[[cat_missing_feature]])
        train_df[cat_missing_feature] = imp.transform(train_df[[cat_missing_feature]])
        test_df[cat_missing_feature] = imp.transform(test_df[[cat_missing_feature]])
    return train_df, test_df

def drop_columns(train_df, test_df):
    drop_columns = ["index"]
    train_df = train_df.drop(columns=drop_columns)
    test_df = test_df.drop(columns=drop_columns)
    return train_df, test_df

def drop_price_outliers(train_df, test_df):
    upper_bound = np.quantile(train_df.price, 0.95)
    train_df = train_df[train_df.price <= upper_bound]
    return train_df, test_df

def drop_insurance_price_outliers(train_df, test_df):
    upper_bound = np.quantile(train_df.insurance_price, 0.99)
    train_df = train_df[train_df.insurance_price <= upper_bound]
    return train_df, test_df

def fill_insurance_price(train_df, test_df):
    train_df.loc[train_df.insurance_price.isna(), "insurance_price"] = train_df.insurance_price.mean()
    return train_df, test_df
    
def fix_registration_year(train_df, test_df):
    train_df.loc[train_df.registration_year < 100, "is_fixed_reg_year"] = 1.0
    train_df.registration_year = train_df.registration_year.apply(lambda y : 2000 + y if y < 21 else y)
    train_df.registration_year = train_df.registration_year.apply(lambda y : 1900 + y if y < 100 else y)
    
    test_df.loc[test_df.registration_year < 100, "is_fixed_reg_year"] = 1.0
    test_df.registration_year = test_df.registration_year.apply(lambda y : 2000 + y if y < 21 else y)
    test_df.registration_year = test_df.registration_year.apply(lambda y : 1900 + y if y < 100 else y)
    return train_df, test_df

def cat_encode(train_df, test_df):
    for cat_feature in cat_features:
        le = LabelEncoder()
        le.fit(pd.concat([train_df, test_df])[cat_feature])
        train_df[cat_feature] = le.transform(train_df[cat_feature])
        test_df[cat_feature] = le.transform(test_df[cat_feature])
        
    return train_df, test_df

def indicate_missing(train_df, test_df):
    for missing_feature in cont_missing_features+cat_missing_features:
        imp = MissingIndicator(missing_values=np.nan)
        imp.fit(pd.concat([train_df, test_df])[[missing_feature]])
        train_df["is_missing_" + missing_feature] = imp.transform(train_df[[missing_feature]])
        test_df["is_missing_" + missing_feature] = imp.transform(test_df[[missing_feature]])
    return train_df, test_df
