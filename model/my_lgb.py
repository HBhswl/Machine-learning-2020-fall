import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from .labels import encoding

def test_one_param(df, label, params):
    iterations = []

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    for i, (train_index, test_index) in enumerate(kf.split(df)):
        x_train, x_test, y_train, y_test = \
            df.loc[train_index, :], df.loc[test_index, :], \
            label.loc[train_index, :], label.loc[test_index, :]

        lgb_train = lgb.Dataset(x_train, y_train["label"].values, silent=True, weight=y_train["weight"].values)
        lgb_test = lgb.Dataset(x_test, y_test["label"].values, silent=True, weight=y_test["weight"].values)

        cv_results = lgb.cv(params, 
                    lgb_train, 
                    num_boost_round=1000, 
                    nfold=5, 
                    stratified=False, 
                    shuffle=True, 
                    early_stopping_rounds=50,
                    seed=0)

        print(cv_results)
        params['num_iterations'] = len(cv_results['multi_logloss-mean'])
        iterations.append(params['num_iterations'])

        bst = lgb.train(params,
            lgb_train,
            valid_sets=lgb_test,
            )
  
    return iterations

def generate_result(df_train, label_train, df_test, params, iterations, using_weight=True):
    result = []

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    for i, (train_index, test_index) in enumerate(kf.split(df_train)):
        x_train, x_test, y_train, y_test = \
            df_train.loc[train_index, :], df_train.loc[test_index, :], \
            label_train.loc[train_index, :], label_train.loc[test_index, :]

        if using_weight:
            lgb_train = lgb.Dataset(x_train, y_train["label"].values, silent=True, weight=y_train["weight"].values)
            lgb_test = lgb.Dataset(x_test, y_test["label"].values, silent=True, weight=y_test["weight"].values)
        else:
            lgb_train = lgb.Dataset(x_train, y_train["label"].values, silent=True)
            lgb_test = lgb.Dataset(x_test, y_test["label"].values, silent=True)
  
        params['num_iterations'] = iterations[i]
        bst = lgb.train(params,
                        lgb_train,
                        valid_sets=lgb_test)

        testye = df_test.copy()
        testye_proba = bst.predict(df_test)
        result.append(testye_proba)
    return np.mean(result, axis=0)

def save_results(keys, df_test, y_pred, df_test_ori):
    y_pred = pd.DataFrame(y_pred, columns=keys)
    y_pred.index = df_test.index
    res = pd.concat([df_test_ori, y_pred], axis=1, join="outer")
    res = res.fillna(0)
    res = res[keys]
    res.to_csv("result-weight-2.csv", index=True, index_label="Id")

def my_lightgbm_model(df_train, df_test, df_test_ori):
    keys, df_train = encoding(df_train)
    df_train.index = np.arange(df_train.shape[0])    

    feature_names = list(df_test.columns)
    feature_names.remove("Id")

    label_name = ["label", "weight"]

    X_train = df_train[feature_names]
    y_train = df_train[label_name]

    X_test = df_test[feature_names]

    params = {
        'objective' : 'multiclass',
        'num_class' : 39,
        'boosting_type' : 'gbdt',
        'learning_rate' : 0.01,

        # 'num_iterations': 200,

        'metric': {'multi_logloss'},
    
        'num_leaves' : 31,
        'max_depth': 6,
        'max_bin' :  256,
        'min_data_in_leaf': 100,

        'feature_fraction' : 0.9,
        'bagging_fraction' : 0.9,
        'bagging_freq': 0,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'min_split_gain': 0.0,
    
        'max_position':10,
        'num_threads': 20,
        'random_state': 0,
        'seed': 0,
        'feature_fraction_seed': 0,
        'bagging_seed': 0,
        'deterministic': True
    }

    # iterations = test_one_param(X_train, y_train, params)
    iterations = [340] * 5
    y_pred = generate_result(X_train, y_train, X_test, params, iterations)
    save_results(keys, X_test, y_pred, df_test_ori)
