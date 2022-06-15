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
train_data = np.load("${TRAIN_NPZ}")

X_train = train_data["X"]
y_train = train_data["y"]
featnames = list(range(X_train.shape[1]))
param_grid = u.read_parameters("${PARAMS_FILE}", "feature_selection", "hsic_lasso")
mode = "categorical" if "${TAG}".split("_")[0] == "categorical" else "regression"

# Run algorithm
############################
hl = HSICLasso()
hl.input(X_train, y_train, featnames)

try:
    if mode == "categorical":
        hl.classification(**param_grid)
    else:
        hl.regression(**param_grid)
except MemoryError:
    u.custom_error(file="scores.npz", content=np.array([]))

selected_feats = np.zeros(shape=(X_train.shape[1],), dtype=bool)
if list(hl.A):
    selected_feats[np.array(hl.A)] = True
# Save selected features
############################
u.save_scores_npz(
    hl.A,
    selected_feats,
    hl.get_index_score(),
    param_grid,
    name="scores_feature_selection_hsic_lasso.npz",
)
# u.save_selected_tsv(hl.A, featnames, param_grid)
