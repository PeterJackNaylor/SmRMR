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
  - selected.npz: contains the featnames of the selected features, their scores and the
    hyperparameters selected by cross-validation
  - selected.tsv: like selected.npz, but in tsv format.
"""

import numpy as np
from pandas import DataFrame
import pymrmr
import utils as u

from sklearn.model_selection import StratifiedKFold

import itertools

def fit_mrmr(X, y, X_val, y_val, list_hyperparameter, mode):
    max_score = -np.inf
    for hp in list_hyperparameter:
        selected_feat = pymrmr.mRMR(X, hp[0], hp[1])

        X_tmp = X[selected_feat]
        X_val_tmp = X_val[selected_feat]
        val_score = u.evaluate_function(X_tmp, y, X_val_tmp, y_val, mode)
        if val_score > max_score:
            max_score = val_score
            best_feats = selected_feat
    return best_feats


def main():
    X, y, featnames, _ = u.read_data("${DATA_NPZ}")
    X = DataFrame(X, columns=featnames)
    u.set_random_state()
    mode = "classification"
    param_grid = u.read_parameters("${PARAMS_FILE}", "feature_selection", "mrmr")
    criteria = param_grid["criteria"]
    num_feat = param_grid["num_features"]
    list_hyperparameter = list(itertools.product(criteria, num_feat))

    folds = u.get_fold("${PARAMS_FILE}")
    selected_feats = DataFrame(index=featnames, columns=["${MODEL.name}"], dtype=int).fillna(0)
    for r in range(int("${REPEATS}")):
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, y_train = X.loc[train_index], y[train_index]
            X_val, y_val = X.loc[test_index], y[test_index]
            
            feats = fit_mrmr(X_train, y_train, X_val, y_val, list_hyperparameter, mode)

            if len(feats):
                selected_feats.loc[feats, "${MODEL.name}"] += 1

    df = DataFrame(selected_feats, columns=["${MODEL.name}"])
    df = df[df["${MODEL.name}"] != 0]
    df.to_csv("${MODEL.name}.csv")

if __name__ == "__main__":
    main()
