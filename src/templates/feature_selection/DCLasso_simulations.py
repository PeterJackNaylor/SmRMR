#!/usr/bin/env python
"""
Input variables:
  - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
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
from operator import itemgetter
import numpy as np
from pandas import DataFrame
from dclasso import DCLasso
import utils as u


def main():
    train_data = np.load("${TRAIN_NPZ}")

    X_train = train_data["X"]
    y_train = train_data["y"]

    val_data = np.load("${VAL_NPZ}")

    X_val = val_data["X"]
    y_val = val_data["y"]

    # test_data = np.load("${TEST_NPZ}")

    # X_test = test_data["X"]
    # y_test = test_data["y"]

    u.set_random_state()

    minimize_val_loss = True
    param_grid = u.read_parameters("${PARAMS_FILE}", "dclasso", "dclasso")

    hyperparameters = {
        "lambda": param_grid["lambda"],
        "learning_rate": param_grid["lr"],
        "penalty": ["${PENALTY}"],
        "optimizer": param_grid["optimizer"],
    }

    dl = DCLasso(
        alpha=param_grid["alpha"],
        measure_stat="${AM}",
        kernel="${KERNEL}",
        hard_alpha=False,
    )

    best_score, best_features, wj, dict_scores = dl.cv_fit(
        X_train,
        y_train,
        X_val,
        y_val,
        param_grid=hyperparameters,
        n1=param_grid["n1"],
        minimize_val_loss=minimize_val_loss,
        max_epoch=param_grid["epoch"],
        refit=True,
    )
    best_hyperparameter = max(dict_scores, key=itemgetter(1))
    selected_feats = np.zeros(shape=(X_train.shape[1],), dtype=bool)

    if best_features:
        selected_feats[np.array(best_features)] = True

    hp_dic = {}
    with open("hyperparameters_selection.dclasso.tsv", "a") as file:
        file.write(f"# score: {best_score}\\n")
        for param in best_hyperparameter.split("_"):
            name, value = param.split("=")
            file.write(f"# {name}: {value}\\n")
            hp_dic[name] = value
        DataFrame({"features": best_features, "weight": wj}).to_csv(
            file, sep="\\t", index=False
        )
    u.save_scores_npz(best_features, selected_feats, wj, hp_dic, "scores_dclasso.npz")


if __name__ == "__main__":
    main()
