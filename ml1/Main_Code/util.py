import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def prep():
    cat_ct = ColumnTransformer(
        [("gill", SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='m'), ['gill-attachment']),
         ("most_freq", SimpleImputer(missing_values=np.nan, strategy='most_frequent'), ['cap-shape', 'cap-surface',
         'cap-color', 'does-bruise-or-bleed', 'gill-spacing', 'gill-color','stem-color', 'has-ring', 'ring-type', 'habitat', 'season'])])
    cat_columns = ['cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
                   'stem-color', 'has-ring', 'ring-type', 'habitat', 'season']
    num_columns = ['cap-diameter', 'stem-height', 'stem-width']
    cat_pipeline = Pipeline([("cat_ct", cat_ct), ("encoder", OneHotEncoder(sparse=False))])
    num_ct = ColumnTransformer([("minmax", MinMaxScaler(), num_columns)])
    preprocessing = ColumnTransformer([('categorical', cat_pipeline, cat_columns), ('numerical', num_ct, num_columns)])
    return preprocessing


def remove_na_columns(pd_dat, threshold=0.5):
    drop_list = []
    for column in pd_dat:
        if pd_dat[column].isna().sum()/pd_dat.shape[0] > threshold:
            drop_list.append(column)
    pd_dat.drop(drop_list, axis=1, inplace=True)
    return pd_dat

def do_grid_search(X, y, classifier, parameter, folds, dateiname):
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameter,
                               cv=folds,
                               return_train_score=True,
                               scoring='precision',
                               n_jobs=-2)
    grid_search.fit(X, y)
    cvres = grid_search.cv_results_
    for mean_train_score, mean_test_score, params in zip(cvres["mean_train_score"], cvres["mean_test_score"], cvres["params"]):
        print(f"{np.sqrt(mean_train_score)}/{np.sqrt(mean_test_score)}, {params}")
    print("="*50)
    res = pd.DataFrame(cvres)
    res.to_csv(dateiname, header=True)
    print(res)