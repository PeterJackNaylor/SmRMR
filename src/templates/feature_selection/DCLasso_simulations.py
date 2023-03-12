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
# from operator import itemgetter
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

    # minimize_val_loss = True
    param_grid = u.read_parameters("${PARAMS_FILE}", "dclasso", "dclasso")

    hyperparameters = {
        "ms": param_grid["measure_stat"],
        "kernel": param_grid["kernel"],
        "lambda": param_grid["lambda"],
        "learning_rate": param_grid["lr"],
        "penalty": ["${PENALTY}"],
        "optimizer": param_grid["optimizer"],
    }
    # n1 = param_grid["n1"]

    dl = DCLasso(
        alpha=param_grid["alpha"],
        measure_stat=None,
        kernel=None,
        hard_alpha=False,
    )

    dl.cv_fit(
        X_train,
        y_train,
        X_val,
        y_val,
        param_grid=hyperparameters,
        n1=param_grid["n1"],
        max_epoch=param_grid["epoch"],
        refit=True,
    )
    best_score = dl.validation_scores[dl.best_run]
    best_hyperparameter = dl.best_hp

    selected_feats = np.zeros(shape=(X_train.shape[1],), dtype=bool)

    if len(dl.alpha_indices_):
        selected_feats[np.array(dl.alpha_indices_)] = True
    best_feat = np.asarray(dl.alpha_indices_)
    wj = np.asarray(dl.wjs_[dl.wjs_ >= dl.t_alpha_])

    hp_dic = {}
    with open("hyperparameters_selection.dclasso.tsv", "a") as file:
        file.write(f"# score: {best_score}\\n")
        for param, val in best_hyperparameter.items():
            if type(val) is dict:
                for item, value in val.items():
                    file.write(f"# {item}: {value}\\n")
                    hp_dic[item] = value
            else:
                file.write(f"# {param}: {val}\\n")
                hp_dic[param] = val

        DataFrame({"features": best_feat, "weight": wj}).to_csv(
            file, sep="\\t", index=False
        )
    u.save_scores_npz(best_feat, selected_feats, wj, hp_dic, "scores_dclasso.npz")


if __name__ == "__main__":
    main()
