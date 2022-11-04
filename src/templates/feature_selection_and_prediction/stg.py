#!/usr/bin/env python
"""
Input variables:
    - TRAIN: path of a numpy array with x.
Output files:
    - selected.npy
"""
import pandas as pd
import numpy as np

import itertools
import torch
from stg import STG
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import utils as u


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
archi = [60, 20]
n_epochs = 1000


def model_train_cv_stg(X, y, hp, device=device):
    learning_rate, sigma, lambda_ = hp

    n_splits = 2
    kf = KFold(n_splits=n_splits)

    score_cv = np.zeros(n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train = X[train_index, :]
        y_train = y[train_index]

        X_val = X[val_index, :]
        y_val = y[val_index]

        model = STG(
            task_type="classification",
            input_dim=X.shape[1],
            output_dim=2,
            hidden_dims=archi,
            activation="tanh",
            optimizer="SGD",
            learning_rate=learning_rate,
            batch_size=X.shape[0],
            feature_selection=True,
            sigma=sigma,
            lam=lambda_,
            random_state=1,
            device=device,
        )

        model.fit(
            X_train,
            y_train,
            nr_epochs=n_epochs,
            valid_X=X_val,
            valid_y=y_val,
            shuffle=True,
            print_interval=200,
        )
        # Prediction (Training)
        feed_dict = {"input": torch.from_numpy(X_val).float()}
        yhat_val = model._model.forward(feed_dict)["prob"].detach().numpy()[:, 1]

        is_auc = 1
        if is_auc:
            score_cv[i] = roc_auc_score(y_val, yhat_val)
        else:
            pos_class = yhat_val > 0.5
            neg_class = yhat_val <= 0.5
            yhat_val[pos_class] = 1
            yhat_val[neg_class] = -1
            score_cv[i] = accuracy_score(y_val, yhat_val)

    return {"pair": hp, "score": score_cv.mean()}


def predict_stg(X, model, mode="classification", device=device):
    feed_dict = {"input": torch.from_numpy(X).float().to(device)}
    yhat = model._model.forward(feed_dict)
    ypred = yhat["pred"].detach()
    if device.type == "cuda":
        ypred = ypred.to("cpu")
    ypred = ypred.numpy()
    if mode == "classification":
        yhat = yhat["prob"].detach()
        if device.type == "cuda":
            yhat = yhat.to("cpu")
        yhat = yhat.numpy()[:, 1]
    else:
        yhat = np.zeros_like(ypred)
    return yhat, ypred


def evalute_stg(X, y, model, mode="classification", device=device):
    # Prediction (Training)
    yhat, ypred = predict_stg(X, model, mode=mode, device=device)
    if mode == "classification":
        score = u.minus_accuracy_score(y, ypred)
        # is_auc = 1
        # if is_auc:
        #     score = roc_auc_score(y, yhat)
        # else:
        #     pos_class = yhat > 0.5
        #     neg_class = yhat <= 0.5
        #     yhat[pos_class] = 1
        #     yhat[neg_class] = -1
        #     score = accuracy_score(y, yhat)
    else:
        # we want to maximise this score
        score = mean_squared_error(y, ypred)

    return score


def stg_model(X, y, learning_rate, sigma, lam, mode="classification", device=device):
    if mode != "classification":
        y = y[:, None]
    model = STG(
        task_type=mode,
        input_dim=X.shape[1],
        output_dim=2 if mode == "classification" else 1,
        hidden_dims=archi,
        activation="tanh",
        optimizer="SGD",
        learning_rate=learning_rate,
        batch_size=X.shape[0],
        feature_selection=True,
        sigma=sigma,
        lam=lam,
        random_state=1,
        device=device,
    )
    # we have to give valid_X and valid_y because it gives an error
    # however, no EARLY STOPPING :-)
    model.fit(
        X, y, nr_epochs=n_epochs, shuffle=True, valid_X=X, valid_y=y, print_interval=200
    )
    return model


mode = "classification" if "${TAG}".split("_")[0] == "categorical" else "regression"

np.random.seed(0)

train_data = np.load("${TRAIN_NPZ}")

X_train = train_data["X"]
y_train = train_data["y"]

val_data = np.load("${VAL_NPZ}")

X_val = val_data["X"]
y_val = val_data["y"]

test_data = np.load("${TEST_NPZ}")

X_test = test_data["X"]
# y_test = test_data["y"]
mode = u.determine_mode("${TAG}")

param_grid = u.read_parameters(
    "${PARAMS_FILE}", "feature_selection_and_prediction", "stg"
)

learning_rates = param_grid["learning_rates"]
sigmas = param_grid["sigmas"]
lambdas = param_grid["lambdas"]

max_score = np.inf
model = None
best_hyperparameter = None
list_hyperparameter = list(itertools.product(learning_rates, sigmas, lambdas))

for hp in list_hyperparameter:
    model = stg_model(X_train, y_train, *hp, mode=mode)
    val_score = evalute_stg(X_val, y_val, model, mode=mode)
    if val_score < max_score:
        best_model = model
        best_hyperparameter = hp
        max_score = val_score

feature_importance = best_model.get_gates(mode="prob")

# filter out unselected genes
selected_index = np.nonzero(feature_importance)[0]
selected_importance = feature_importance[selected_index]
y_proba_test, y_pred_test = predict_stg(X_test, best_model, mode=mode, device=device)

u.save_preds_npz(y_pred_test, best_hyperparameter)
u.save_proba_npz(y_proba_test, best_hyperparameter)

hp_dic = {
    "learning_rate": best_hyperparameter[0],
    "sigma": best_hyperparameter[1],
    "lambda": best_hyperparameter[2],
}

with open("scored.stg.tsv", "a") as f:
    for key, item in hp_dic.items():
        f.write(f"# {key}: {item}\\n")

    pd.DataFrame({"features": selected_index, "weight": selected_importance}).to_csv(
        f, sep="\\t", index=False
    )

selected_feats = np.zeros(shape=(X_train.shape[1],), dtype=bool)
if list(selected_index):
    selected_feats[np.array(selected_index)] = True

u.save_scores_npz(
    selected_index, selected_feats, selected_importance, hp_dic, "scores_stg.npz"
)
