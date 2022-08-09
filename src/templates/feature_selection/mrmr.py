#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing the train set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - TEST_NPZ: path to a .npz file containing the test set. It must contain three
    elements: an X matrix, a y vector, and a featnames vector (optional)
  - PARAMS_JSON: path to a json file with the hyperparameters
    - n_nonzero_coefs
Output files:
  - y_proba.npz: predictions on the test set.
  - scores.npz: contains the featnames, wether each feature was selected, their scores
    and the hyperparameters selected by cross-validation
  - scores.tsv: like scores.npz, but in tsv format
"""
import numpy as np
import pandas as pd
import itertools

import pymrmr
import utils as u

# Prepare data
############################
X, y, featnames, selected = u.read_data("${TRAIN_NPZ}")
featnames = [str(el) for el in list(featnames)]
X = pd.DataFrame(X, columns=featnames)
ds = np.hstack((np.expand_dims(y, axis=1), X))
X_val, y_val, _, _ = u.read_data("${VAL_NPZ}")
X_val = pd.DataFrame(X_val, columns=featnames)

test_data = np.load("${TEST_NPZ}")

X_test = test_data["X"]

mode = u.determine_mode("${TAG}")

# Run mRMR
############################
samples, features = X.shape
param_grid = u.read_parameters("${PARAMS_FILE}", "feature_selection", "mrmr")

criteria = param_grid["criteria"]
num_feat = param_grid["num_features"]

max_score = np.inf
model = None
best_hyperparameter = None
best_feats = None
list_hyperparameter = list(itertools.product(criteria, num_feat))

for hp in list_hyperparameter:
    selected_feat = pymrmr.mRMR(X, hp[0], hp[1])

    X_tmp = X[selected_feat]
    X_val_tmp = X_val[selected_feat]
    val_score = u.evaluate_function(X_tmp, y, X_val_tmp, y_val, mode)
    if val_score < max_score:
        best_model = model
        best_hyperparameter = hp
        max_score = val_score
        best_feats = selected_feat


# Get selected features
############################
best_feats = np.array([int(el) for el in best_feats])

hp_dic = {
    "critera": hp[0],
    "num_feats": hp[1],
}

selected_feat = np.zeros_like(featnames, dtype="bool")
selected_feat[best_feats] = True
scores = np.zeros_like(featnames, dtype="float")
u.save_scores_npz(
    best_feats, selected_feat, scores, hp_dic, name="scores_feature_selection_mrmr.npz"
)


with open("scored.mrmr.tsv", "a") as f:
    for key, item in hp_dic.items():
        f.write(f"# {key}: {item}\\n")

    pd.DataFrame({"features": best_feats}).to_csv(f, sep="\\t", index=False)
