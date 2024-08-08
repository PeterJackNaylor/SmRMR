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
from dclasso import DCLasso
import utils as u
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold

mapping = {}

def main():
    X, y, featnames, _ = u.read_data("${DATA_NPZ}")
    X = DataFrame(X, columns=featnames)

    u.set_random_state()

    param_grid = u.read_parameters("${PARAMS_FILE}", "dclasso", "dclasso")
    folds = u.get_fold("${PARAMS_FILE}")

    selected_feats = DataFrame(0, index=featnames, columns=["${METHOD}"], dtype=int).fillna(0)
    for r in range(int("${REPEATS}")):
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, y_train = X.loc[train_index], y[train_index]
            X_val, y_val = X.loc[test_index], y[test_index]
            dl = DCLasso(
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

            if dl.t_alpha_ != np.inf:
                selected_feats.loc[featnames[np.array(dl.alpha_indices_)], "${METHOD}"] += 1
          
    df = DataFrame(selected_feats, columns=["${METHOD}"])
    df = df[df["${METHOD}"] != 0]
    df.to_csv("${METHOD}.csv")

if __name__ == "__main__":
    main()
