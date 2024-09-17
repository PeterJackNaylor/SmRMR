#!/usr/bin/env python
"""
Input variables:
  - DATA_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
  - PARAMS_FILE: path to a json file with the hyperparameters
    - num_feat
    - weight_decay
    - penalisation
    - optimizer
    - learning rate
    - covars
Output files:
  - selected_features.csv: contains the featnames of the 
  selected features and how many times they have been selected
"""

import numpy as np
from pandas import DataFrame
from pyHSICLasso import HSICLasso
import utils as u
from sklearn.model_selection import StratifiedKFold

def fit_hsic_lasso(X, y, X_val, y_val, featnames, num_feat, mode):
    hl = HSICLasso()
    hl.input(X.values, y, featname=featnames)
    p = X.shape[1]
    max_score = np.inf
    for nfeat in num_feat:
        if mode == "categorical":
            hl.classification(nfeat)
        else:
            hl.regression(nfeat)
        selected_feats = np.zeros(shape=(p,), dtype=bool)
        if list(hl.A):
            selected_feats[np.array(hl.A)] = True
        X_tmp = X.loc[:, selected_feats]
        X_val_tmp = X_val.loc[:, selected_feats]
        val_score = u.evaluate_function(X_tmp, y, X_val_tmp, y_val, mode)
        if val_score < max_score:
            max_score = val_score
            best_feats = selected_feats
    return best_feats

def main():
    X, y, featnames, _ = u.read_data("${DATA_NPZ}")
    X = DataFrame(X, columns=featnames)

    u.set_random_state()
    param_grid = u.read_parameters("${PARAMS_FILE}", "feature_selection", "hsic_lasso")
    mode = "classification"
    num_feat = param_grid["num_features"]
    folds = u.get_fold("${PARAMS_FILE}")
    selected_feats = DataFrame(index=featnames, columns=["${MODEL.name}"], dtype=int).fillna(0)
    scores = []
    for r in range(int("${REPEATS}")):
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_, y_ = X.loc[train_index], y[train_index]
            X_test, y_test = X.loc[test_index], y[test_index]

            choosen_feats  = []
            for j, (train_index2, val_index) in enumerate(skf.split(X_, y_)):
                X_train, y_train = X.loc[train_index2], y[train_index2]
                X_val, y_val = X.loc[val_index], y[val_index]
                try:
                    feats = fit_hsic_lasso(X_train, y_train, X_val, y_val,  featnames, num_feat, mode)
                    feats = np.arange(feats.shape[0])[feats]
                    if len(feats):
                        selected_feats.loc[feats.astype(str), "${MODEL.name}"] += 1

                    choosen_feats.append(feats.astype(str))
                except:
                    pass
            if choosen_feats:
                union_list = u.union_lists(*choosen_feats)
                X_train_tmp = X_.loc[:, union_list]
                X_test_tmp = X_test.loc[:, union_list]
                test_score = u.evaluate_function(X_train_tmp, y_, X_test_tmp, y_test, mode)
                scores.append(1 - test_score)

    df = DataFrame(selected_feats, columns=["${MODEL.name}"])
    df = df[df["${MODEL.name}"] != 0]
    df.to_csv("${MODEL.name}.csv")
    file = open('${MODEL.name}_scores.txt','w')
    for s in scores:
        file.write(f"{s}\\n")
    file.close()
if __name__ == "__main__":
    main()
