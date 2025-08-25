#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
  - PARAMS_FILE: path to a json file with the hyperparameters
    - am
    - kernel
Output files:
  - selected.npz: contains the featnames of the selected features, their scores and the
    hyperparameters selected by cross-validation
  - selected.tsv: like selected.npz, but in tsv format.
"""
import numpy as np
from jax import random
from sklearn.utils.validation import check_X_y
from pandas import DataFrame

from smrmr import smrmr
from smrmr.smrmr_class import alpha_threshold
import utils as u
from smrmr.utils import (
    knock_off_check_parameters,
    get_equi_features,
    generate_random_sets,
)

u.set_random_state()

X, y, featnames, selected = u.read_data("${TRAIN_NPZ}")
X_val, y_val, _, _ = u.read_data("${VAL_NPZ}")
p = X.shape[1]


param_grid = u.read_parameters("${PARAMS_FILE}", "feature_selection", "hsic_lasso")

# Run algorithm
############################
# Algorithm parameters
seed = 42
n1 = param_grid["n1"]
alpha = param_grid["alpha"]


def main():
    # Read data
    ############################
    train_data = np.load("${TRAIN_NPZ}")

    X_train = train_data["X"]
    y_train = train_data["y"]
    dl = smrmr(alpha=param_grid["alpha"], measure_stat="${AM}", kernel="${KERNEL}")

    key = random.PRNGKey(seed)

    X, y = check_X_y(X_train, y_train)
    X = np.asarray(X)
    n, p = X.shape

    (ny,) = y.shape
    assert n == ny
    y = np.asarray(y)

    # we have to prescreen and split if needed
    stop, screening, msg, d = knock_off_check_parameters(n, p, n1, None)
    if screening:
        # 1 - pre-screening by only checking the marginals
        s1, s2 = generate_random_sets(n, n1, key)
        X1, y1 = X[s1, :], y[s1]
        X2, y2 = X[s2, :], y[s2]
        screened_indices = dl.marginal_screen(X1, y1, d)
        X1 = X1[:, screened_indices]
        X2 = X2[:, screened_indices]

    Xhat = get_equi_features(X2, key)

    Xs = np.concatenate([X2, Xhat], axis=1)
    beta_ = dl._compute_assoc(Xs, y2, **dl.ms_kwargs)

    wj = beta_[:d] - beta_[d:]

    alpha_thres = alpha_threshold(
        alpha,
        wj,
        screened_indices,
        hard_alpha=False,
        alpha_increase=0.05,
        verbose=False,
    )

    alpha_indices_ = alpha_thres[0]
    t_alpha_ = alpha_thres[1]
    n_features_out_ = alpha_thres[2]

    selected_feats = np.zeros(shape=(X_train.shape[1],), dtype=bool)

    if alpha_indices_:
        selected_feats[np.array(alpha_indices_)] = True

    hp_dic = {
        "alpha": alpha,
        "n1": n1,
        "d": d,
        "n_features_out": n_features_out_,
        "t_alpha": t_alpha_,
        "measure_stat": "${AM}",
        "kernel": "${KERNEL}",
    }
    with open("hyperparameters_selection.knockoff.tsv", "a") as file:
        for key, value in hp_dic.items():
            file.write(f"# {key}: {value}\\n")
        DataFrame({"features": alpha_indices_, "weight": wj}).to_csv(
            file, sep="\\t", index=False
        )
    u.save_scores_npz(alpha_indices_, selected_feats, wj, hp_dic, "scores_knockoff.npz")


if __name__ == "__main__":
    main()
