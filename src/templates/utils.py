import random
import sys
import traceback

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from scipy.sparse import load_npz

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score


# Input functions
###########################
def read_data(data_npz: str, selected_npz: str = ""):
    data = np.load(data_npz, allow_pickle=True)

    X = data["X"]
    y = data["y"]

    if "featnames" in data.keys():
        featnames = data["featnames"]
    else:
        featnames = np.arange(X.shape[1])

    if selected_npz != "":
        featnames = np.load(selected_npz)["featnames"]
        selected = np.load(selected_npz)["selected"]

        if not sum(selected):
            custom_error()
    else:
        selected = np.zeros(X.shape[0], dtype=bool)
    return X, y, featnames, selected


def read_adjacency(A_npz: str):

    return load_npz(A_npz)


def read_parameters(params_yaml: str, algo_type: str, algo_name: str) -> dict:

    f = open(params_yaml)

    for x in yaml.load(f, Loader=yaml.Loader)[algo_type]:
        if x["name"] == algo_name:
            return x["parameters"]

    return {}


# Output functions
##########################
def update_save_scores_npz(
    featnames: npt.ArrayLike,
    selected: npt.ArrayLike,
    scores: npt.ArrayLike = None,
    hyperparams: dict = None,
    name: str = "scores.npz",
    new_name: str = "scores_new.npz",
):
    old_scores_file = np.load(name, allow_pickle=True)
    old_featnames = old_scores_file["featnames"]
    old_scores = old_scores_file["scores"]
    old_selected = old_scores_file["selected"]
    old_hyperparams = old_scores_file["hyperparams"]
    np.savez(
        new_name,
        featnames=featnames,
        scores=sanitize_vector(scores),
        selected=sanitize_vector(selected),
        hyperparams=hyperparams,
        old_featnames=old_featnames,
        old_scores=old_scores,
        old_selected=old_selected,
        old_hyperparams=old_hyperparams,
    )


def save_scores_npz(
    featnames: npt.ArrayLike,
    selected: npt.ArrayLike,
    scores: npt.ArrayLike = None,
    hyperparams: dict = None,
    name: str = "scores.npz",
):
    np.savez(
        name,
        featnames=featnames,
        scores=sanitize_vector(scores),
        selected=sanitize_vector(selected),
        hyperparams=hyperparams,
    )


def save_scores_tsv(
    featnames: npt.ArrayLike,
    selected: npt.ArrayLike,
    scores: npt.ArrayLike = None,
    hyperparams: dict = {},
):
    features_dict = {"feature": featnames, "weights": sanitize_vector(scores)}
    if scores is not None:
        features_dict["score"] = sanitize_vector(scores)

    with open("scores.tsv", "a") as FILE:
        for key, value in hyperparams.items():
            FILE.write("# {}: {}\\n".format(key, value))
        pd.DataFrame(features_dict).to_csv(FILE, sep="\t", index=False)


def save_preds_npz(preds: npt.ArrayLike = None, hyperparams: dict = None):
    np.savez("y_pred.npz", preds=sanitize_vector(preds), hyperparams=hyperparams)


def save_proba_npz(proba: npt.ArrayLike = None, hyperparams: dict = None):
    np.savez("y_proba.npz", proba=sanitize_vector(proba), hyperparams=hyperparams)


def save_analysis_tsv(**kwargs):
    metrics_dict = locals()["kwargs"]
    with open("performance.tsv", "w", newline="") as FILE:
        pd.DataFrame(metrics_dict).to_csv(FILE, sep="\t", index=False)


# Other functions
##########################
def set_random_state(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def custom_error(error: int = 77, file: str = None, content=None):
    traceback.print_exc()
    np.save(file, content)
    sys.exit(error)


def sanitize_vector(x: npt.ArrayLike):
    if x is not None:
        x = np.array(x)
        x = x.flatten()

    return x


def minus_accuracy_score(y, yhat):
    return 1 - accuracy_score(y, yhat)


def determine_mode(name: str):
    if "categorical" in name:
        return "classification"
    elif "linear" in name:
        return "regression"
    else:
        raise ValueError(f"Unknown model name: {name}")


def model_eval(mode):
    if mode == "classification":
        model = LogisticRegression()
        eval_func = minus_accuracy_score
    else:
        model = LinearRegression()
        eval_func = mean_squared_error
    return model, eval_func


def evaluate_function(X, y, X_val, y_val, mode="classification"):
    model, eval_func = model_eval(mode)
    model.fit(X, y)
    return eval_func(model.predict(X_val), y_val)
