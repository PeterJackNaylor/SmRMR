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
import sys
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


def fit(X, y, X_val, y_val, mode, params):
    p = X.shape[1]
    param_grid = u.read_parameters(params, "feature_selection", "hsic_lasso")

    num_feat = param_grid["num_features"]
    max_score = np.inf
    best_A = None
    # best_hyperparameter = None
    best_feats = None
    # best_scores = None
    best_model = None

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
            best_A = hl.A
            # best_hyperparameter = {"num_feat": nfeat}
            max_score = val_score
            best_feats = selected_feats.copy()
            # best_scores = hl.get_index_score()
            best_model = hl
    return best_model, np.array(best_A), best_feats
    # Save selected features
    ############################
    # u.save_scores_npz(
    #     best_A,
    #     best_feats,
    #     best_scores,
    #     best_hyperparameter,
    #     name="scores_feature_selection_hsic_lasso.npz",
    # )


# u.save_selected_tsv(hl.A, featnames, param_grid)


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
        results.loc[i, "n"] = n_p.split("-")[0]
        results.loc[i, "p"] = n_p.split("-")[1]

        f_causal = f.replace("simulation", "causal")
        f_validation = f.replace("simulation", "simulation_val")
        f_test = f.replace("simulation", "simulation_test").replace(n_p, "*")
        f_test = glob(f_test)[0]

        X, y, featnames, selected = u.read_data(f)
        X_val, y_val, _, _ = u.read_data(f_validation)
        mode = u.determine_mode(data_name)
        model_fs, best_features, selected_features = fit(
            X, y, X_val, y_val, mode, PARAMS_FILE
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
    results["METHOD"] = "HSICLasso"
    results.to_csv("results_hsic_lasso.csv")
