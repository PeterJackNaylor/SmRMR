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
from smrmr import smrmr
import utils as u
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold

mapping = {}

def main():
    X, y, featnames, _ = u.read_data("${DATA_NPZ}")
    X = DataFrame(X, columns=featnames)

    u.set_random_state()

    param_grid = u.read_parameters("${PARAMS_FILE}", "smrmr", "smrmr")
    mode = "classification"
    folds = u.get_fold("${PARAMS_FILE}")

    selected_feats = DataFrame(0, index=featnames, columns=["${METHOD}"], dtype=int).fillna(0)
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
                  dl = smrmr(
                      alpha=param_grid["alpha"],
                      measure_stat="${MS}",
                      kernel="${KERNEL}",
                      hard_alpha=False,
                  )
                  train_l, val_l = dl.cv_fit(
                      X_train,
                      y_train,
                      X_val,
                      y_val,
                      "${PENALTY}",
                      param_grid["lambda"],
                      n1=param_grid["n1"],
                      pen_kwargs=dict(a=3.7, b=3.5),
                  )
                  
                  feats = featnames[np.array(dl.alpha_indices_)]
                  
                  if len(feats):
                      selected_feats.loc[feats, "${METHOD}"] += 1
                  choosen_feats.append(feats)
                except:
                    pass
            if choosen_feats:
                union_list = u.union_lists(*choosen_feats)
                X_train_tmp = X_.loc[:, union_list]
                X_test_tmp = X_test.loc[:, union_list]
                test_score = u.evaluate_function(X_train_tmp, y_, X_test_tmp, y_test, mode)
                scores.append(1 - test_score)

    df = DataFrame(selected_feats, columns=["${METHOD}"])
    df = df[df["${METHOD}"] != 0]
    df.to_csv("${METHOD}.csv")

    file = open('${METHOD}_scores.txt','w')
    for s in scores:
        file.write(f"{s}\\n")
    file.close()
if __name__ == "__main__":
    main()
