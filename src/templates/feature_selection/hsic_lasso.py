#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
  - PARAMS_FILE: path to a json file with the hyperparameters
    - num_feat
    - B
    - M
    - covars
Output files:
  - selected.npz: contains the featnames of the selected features, their scores and the
    hyperparameters selected by cross-validation
  - selected.tsv: like selected.npz, but in tsv format.
"""
import numpy as np
from pyHSICLasso import HSICLasso

import utils as u

u.set_random_state()

# Read data
############################
X, y, featnames, selected = u.read_data("${TRAIN_NPZ}")
X_val, y_val, _, _ = u.read_data("${VAL_NPZ}")
p = X.shape[1]


param_grid = u.read_parameters("${PARAMS_FILE}", "feature_selection", "hsic_lasso")

num_feat = param_grid["num_features"]
max_score = np.inf
best_A = None
best_hyperparameter = None
best_feats = None
best_scores = None

mode = u.determine_mode("${TAG}")

# Run algorithm
############################
hl = HSICLasso()
hl.input(X, y, featname=featnames)

for nfeat in num_feat:
    if mode == "categorical":
        hl.classification(nfeat)
    else:
        hl.regression(nfeat)
    selected_feats = np.zeros(shape=(p,), dtype=bool)
    if list(hl.A):
        selected_feats[np.array(hl.A)] = True
    X_tmp = X[:, selected_feats]
    X_val_tmp = X_val[:, selected_feats]
    val_score = u.evaluate_function(X_tmp, y, X_val_tmp, y_val, mode)
    if val_score < max_score:
        best_A = (hl.A,)
        best_hyperparameter = {"num_feat": nfeat}
        max_score = val_score
        best_feats = selected_feats
        best_scores = hl.get_index_score()
# Save selected features
############################
u.save_scores_npz(
    best_A,
    best_feats,
    best_scores,
    best_hyperparameter,
    name="scores_feature_selection_hsic_lasso.npz",
)
# u.save_selected_tsv(hl.A, featnames, param_grid)
