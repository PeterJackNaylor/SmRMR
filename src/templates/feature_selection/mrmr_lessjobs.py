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
import sys
import pymrmr
import utils as u
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from base.sklearn import SklearnModel
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from pandas import DataFrame

u.set_random_state()

# Read data
############################


def tpr_fpr(causal, selected):
    if len(causal):
        tn, fp, fn, tp = confusion_matrix(causal, selected, labels=[0, 1]).ravel()
        if tp + fn != 0:
            tpr = tp / (tp + fn)
        else:
            tpr = "NA"

        if fp + tn != 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0

        if tp + fp != 0:
            fdr = fp / (tp + fp)
        else:
            fdr = 0

    else:
        tpr = fpr = fdr = "NA"
    return tpr, fpr, fdr


class PreBase(SklearnModel):
    def train_validate(self, X, y, featnames, selected, X_val, y_val, params_file):

        # X, y, featnames, selected = u.read_data(train_npz, scores_npz)
        self.model_input_features = selected
        # featnames = np.squeeze(featnames)
        if len(featnames.shape) >= 2:
            featnames = featnames[0]
            # if featnames.shape[1] > 1:
            #     featnames = np.squeeze(featnames)
            # else:
            #     featnames =
        X = X[:, selected]
        X_val = X_val[:, selected]
        param_grid = u.read_parameters(params_file, self.config_name, self.name)
        X_join = np.concatenate((X, X_val), axis=0)
        y_join = np.concatenate((y, y_val), axis=0)
        split_index = np.concatenate(
            (np.zeros(len(y), dtype=int) - 1, np.zeros(len(y_val), dtype=int)), axis=0
        )
        pds = PredefinedSplit(test_fold=split_index)

        self.clf = GridSearchCV(self.model, param_grid, cv=pds, scoring=self.scoring)
        self.clf.fit(X_join, y_join)

        self.best_hyperparams = {k: self.clf.best_params_[k] for k in param_grid.keys()}

        scores = self.score_features()
        scores = u.sanitize_vector(scores)
        self_selected = self.select_features(scores)

        featnames = featnames[self_selected]
        selected_feats = np.zeros(shape=(selected.shape[0],), dtype=bool)

        if list(featnames):
            selected_feats[np.array(featnames)] = True
        self.select_features = selected_feats


class LassoModel(PreBase):
    def __init__(self) -> None:
        lasso = Lasso()
        super().__init__(lasso, "regression", "prediction", "lasso")

    def score_features(self):
        return self.clf.best_estimator_.coef_

    def select_features(self, scores):
        return scores != 0


class RandomForestModel(PreBase):
    def __init__(self) -> None:
        rf = RandomForestClassifier()
        super().__init__(rf, "classification", "prediction", "random_forest")

    def score_features(self):
        return self.clf.best_estimator_.feature_importances_

    def select_features(self, scores):
        return scores != 0


def fit(X, y, X_val, y_val, featnames, mode):
    # Prepare data
    ############################
    # X, y, featnames, selected = u.read_data("${TRAIN_NPZ}")
    featnames = [str(el) for el in list(featnames)]
    X = pd.DataFrame(X, columns=featnames)
    # ds = np.hstack((np.expand_dims(y, axis=1), X))
    # X_val, y_val, _, _ = u.read_data("${VAL_NPZ}")
    X_val = pd.DataFrame(X_val, columns=featnames)

    # Run mRMR
    ############################
    samples, features = X.shape
    param_grid = u.read_parameters(PARAMS_FILE, "feature_selection", "mrmr")

    criteria = param_grid["criteria"]
    num_feat = param_grid["num_features"]

    max_score = np.inf
    model = None
    # best_hyperparameter = None
    best_feats = None
    list_hyperparameter = list(itertools.product(criteria, num_feat))

    for hp in list_hyperparameter:
        selected_feat = pymrmr.mRMR(X, hp[0], hp[1])

        X_tmp = X[selected_feat]
        X_val_tmp = X_val[selected_feat]
        val_score = u.evaluate_function(X_tmp, y, X_val_tmp, y_val, mode)
        if val_score < max_score:
            best_model = model
            # best_hyperparameter = hp
            max_score = val_score
            best_feats = selected_feat

    # Get selected features
    ############################
    best_feats = np.array([int(el) for el in best_feats])

    # hp_dic = {
    #     "critera": hp[0],
    #     "num_feats": hp[1],
    # }

    selected_feat = np.zeros_like(featnames, dtype="bool")
    selected_feat[best_feats] = True
    return best_model, best_feats, selected_feat
    # scores = np.zeros_like(featnames, dtype="float")
    # u.save_scores_npz(
    #     best_feats, selected_feat, scores, hp_dic,
    # name="scores_feature_selection_mrmr.npz"
    # )

    # with open("scored.mrmr.tsv", "a") as f:
    #     for key, item in hp_dic.items():
    #         f.write(f"# {key}: {item}\\n")

    #     pd.DataFrame({"features": best_feats}).to_csv(f, sep="\\t", index=False)


if __name__ == "__main__":
    # MS = sys.argv[1]
    # KERNEL = sys.argv[2]
    # PENALTY = sys.argv[3]
    # PARAMS_FILE = sys.argv[4]
    PARAMS_FILE = sys.argv[1]

    files = glob("simulation__*.npz")
    results = DataFrame(
        index=range(0, len(files)),
        columns=["name", "n", "p", "rep", "mse", "acc", "tpr", "fpr", "fdr"],
    )

    for i, f in enumerate(files):

        data_name = f.split("__")[1]
        results.loc[i, "name"] = data_name.split("-")[0]
        results.loc[i, "rep"] = f.split("__")[-1].split(".")[0]
        n_p = "-".join(data_name.split("-")[1:])
        p = n_p.split("-")[1]
        results.loc[i, "n"] = n_p.split("-")[0]
        results.loc[i, "p"] = p

        f_causal = f.replace("simulation", "causal")
        f_validation = glob(f"simulation_val__{data_name}-*-{p}__1.npz")[0]
        f_test = glob(f"simulation_test__{data_name}-*-{p}__1.npz")[0]

        X, y, featnames, selected = u.read_data(f)
        X_val, y_val, _, _ = u.read_data(f_validation)
        mode = u.determine_mode(data_name)
        model_fs, best_features, selected_features = fit(
            X, y, X_val, y_val, featnames, mode
        )
        if mode == "classification":
            model = RandomForestModel()
        else:
            model = LassoModel()

        model.train_validate(
            X, y, best_features, selected_features, X_val, y_val, PARAMS_FILE
        )

        test_set = np.load(f_test)
        X_test, y_test = test_set["X"], test_set["y"]

        if mode == "classification":
            y_proba = model.clf.predict_proba(X_test[:, model.model_input_features])
            y_pred = y_proba.argmax(axis=1)
            results.loc[i, "acc"] = accuracy_score(y_test, y_pred)
        else:
            y_pred = model.clf.predict(X_test[:, model.model_input_features])
            results.loc[i, "mse"] = mean_squared_error(y_test, y_pred)

        causal = np.load(f_causal)["selected"]
        tpr, fpr, fdr = tpr_fpr(causal, selected_features)
        results.loc[i, "tpr"] = tpr
        results.loc[i, "fpr"] = fpr
        results.loc[i, "fdr"] = fdr
    results["METHOD"] = "MRMR"
    results.to_csv("results_MRMR.csv")
