#!/usr/bin/env python
"""
Input variables:
    - TRAIN_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional)
    - VAL_NPZ: path to a .npz file containing three elements: an X matrix, a y vector,
    and a featnames vector (optional). Used for the validation set
    - CAUSAL_NPZ: path to a .npz file containing true covariates
    - AM: string, association measure to use
    - KERNEL: string, kernel to be used
Output files:
    - performance.tsv: path to a single-line tsv with the TSV results.
"""
import uuid
from functools import partial
import yaml
import numpy as np
from jax import random
import jax.numpy as jnp
from sklearn.utils.validation import check_X_y

# from scipy.sparse.linalg import eigsh

# try:
#     from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
# except ImportError:
#     from scipy.sparse.linalg import ArpackNoConvergence

from dclasso import DCLasso, pic_penalty
from dclasso.dc_lasso import loss

import utils as u
from lambda_control.lambda_control_utils import (
    build_iterator,
    build_ms_kern_iterator,
    # fdr,
    perform_alpha_computations,
    perform_optimisation_with_parameters,
)

key = random.PRNGKey(42)
Cst = 1.5
max_epoch = 1500
eps_stop = 1e-8
opt_kwargs = {
    "init_value": 0.001,
    "transition_steps": 100,
    "decay_rate": 0.99,
}

am_kernels = ["HSIC", "cMMD"]

# Read data
############################
X, y, featnames, _ = u.read_data("${TRAIN_NPZ}")
X, y = check_X_y(X, y)
X = np.asarray(X)
n, p = X.shape
(ny,) = y.shape
assert n == ny
p = X.shape[1]

X_val, y_val, _, _ = u.read_data("${VAL_NPZ}")
X_val, y_val = check_X_y(X_val, y_val)
X_val = np.asarray(X_val)
y_val = jnp.array(y_val)

causal_feats = np.load("${CAUSAL_NPZ}")
causal_feats = list(causal_feats["featnames"][causal_feats["selected"]])

conservative = len(causal_feats) >= 5


# Hyper-parameters
############################

# hyper-parameters in the for loop
param_grid = {}
f = open("${PARAMS_FILE}")
param_grid = yaml.load(f, Loader=yaml.Loader)

ms_list = param_grid["measure_stat"]
kernel_list = param_grid["kernel"]
penalty_list = param_grid["penalty"]
optimizer_list = param_grid["optimizer"]
lambda_list = param_grid["lambda"]
n1 = param_grid["n1"]

ms_kernel_generator = build_ms_kern_iterator(ms_list, kernel_list)

# File parameters
alpha_list = np.arange(0.1, 1, 0.05)
loss_trains = []
loss_valids = []
hp_parameter = []
fdr_selected_dict = {}

# Rs = []
# N1s = []

# fdr_alpha = []
# selected_variables = []
# penalties_parameter = []
# optimizers__parameter = []
# lambda_parameter = []
# alpha_parameter = []
# default parameters

# Process

# One round to fit and get parameters
for ms, kernel in ms_kernel_generator:
    iterable = build_iterator(penalty_list, optimizer_list, lambda_list)
    penalty_init, optimizer_init, lambda_init = next(iterable)
    penalty_kwargs = {"name": penalty_init, "lamb": lambda_init}

    dl = DCLasso(alpha=alpha_list[0], measure_stat=ms, kernel=kernel)

    dl.fit(
        X,
        y,
        n1=n1,
        max_epoch=max_epoch,
        penalty_kwargs=penalty_kwargs,
        optimizer=optimizer_list[0],
        conservative=conservative,
    )
    loss_fn = partial(
        loss,
        Dxy=dl.Dxy,
        Dxx=dl.Dxx,
        penalty_func=pic_penalty({"name": "None"}),
    )

    loss_train__ = float(loss_fn(dl.beta_))
    # selected_ = list(np.array(dl.alpha_indices_))
    # fdr_ = fdr(causal_feats, selected_)

    # R__ = float(Cst / penalty_kwargs["lamb"] * pic_penalty(penalty_kwargs)(dl.beta_))
    # N1__ = np.abs(dl.beta_).sum()

    # Compute validation loss, but only compute X_val, Xhat and Dxx/Dxy once
    d = len(dl.screen_indices_)
    X_val_sub = X_val[:, dl.screen_indices_]

    if "precompute" in dl.ms_kwargs.keys():
        del dl.ms_kwargs["precompute"]

    Dxy_val = dl._compute_assoc(X_val_sub, y_val, **dl.ms_kwargs)
    Dxx_val = dl._compute_assoc(X_val_sub, **dl.ms_kwargs)
    dl.Dxy_val = Dxy_val
    dl.Dxx_val = Dxx_val

    # Compute validation loss fn
    loss_fn = partial(
        loss, Dxy=Dxy_val, Dxx=Dxx_val, penalty_func=pic_penalty(penalty_kwargs)
    )
    loss_valid__ = float(loss_fn(dl.beta_[:d]))

    fdr__, selected__ = perform_alpha_computations(
        alpha_list, dl.wjs_, dl.screen_indices_, causal_feats, conservative
    )

    hp = (penalty_init, optimizer_init, lambda_init, ms, kernel)
    hp_parameter += [hp]
    loss_trains += [loss_train__]  # * len(alpha_list)
    loss_valids += [loss_valid__]  # * len(alpha_list)
    fdr_selected_dict[hp] = (fdr__, selected__)
    # penalties_parameter += [] # * len(alpha_list)
    # optimizers__parameter += [] # * len(alpha_list)
    # lambda_parameter += [] # * len(alpha_list)
    # alpha_parameter += list(alpha_list)
    # Rs += [R__] * len(alpha_list)
    # N1s += [N1__] * len(alpha_list)
    # fdr_alpha += [fdr_] + fdr__
    # selected_variables += [selected_] + selected__

    # Loop over parameters

    for pen, opt, lam in iterable:
        penalty_kwargs = {"name": pen, "lamb": lam}
        (
            fdr__,
            selected__,
            loss_train__,
            loss_valid__,
            R__,
            N1__,
        ) = perform_optimisation_with_parameters(
            dl,
            pen,
            opt,
            lam,
            alpha_list,
            d,
            causal_feats,
            key,
            max_epoch,
            eps_stop,
            opt_kwargs,
            Cst,
            penalty_kwargs,
            conservative,
        )
        hp = (pen, opt, lam, ms, kernel)
        hp_parameter += [hp]
        loss_trains += [loss_train__]  # * len(alpha_list)
        loss_valids += [loss_valid__]  # * len(alpha_list)
        fdr_selected_dict[hp] = (fdr__, selected__)
        # Rs += [R__] * len(alpha_list)
        # N1s += [N1__] * len(alpha_list)
        # fdr_alpha += fdr__
        # selected_variables += selected__
        # penalties_parameter += [pen] # * len(alpha_list)
        # optimizers__parameter += [opt] # * len(alpha_list)
        # lambda_parameter += [lam] # * len(alpha_list)
        # alpha_parameter += list(alpha_list)


# Get the best model, i.e. minimizes the validation loss
idx_star = np.argmin(loss_valids)
hp_star = hp_parameter[idx_star]
loss_train_star = loss_trains[idx_star]
loss_valid_star = loss_valids[idx_star]
fdr_star, selected_star = fdr_selected_dict[hp_star]

# Some theoritical results that depends only on the data
# try:
#     alpha2 = float(eigsh(np.array(dl.Dxx), k=1, which="SA")[0].squeeze())
# except ArpackNoConvergence:
#     alpha2 = 0
# Save results
############################

simu, _ = "${TAG}".split("(")
run_id = str(uuid.uuid4())

u.save_analysis_tsv(
    run=simu,
    run_id=run_id,
    n=n,
    p=p,
    MS=hp_star[3],
    kernel=hp_star[4],
    penalty=hp_star[0],
    optimizer=hp_star[1],
    metric="fdr",
    lamb=hp_star[2],
    alpha=alpha_list,
    value=fdr_star,
    selected=selected_star,
    loss_train=loss_train_star,
    loss_valid=loss_valid_star,
    # alpha2=alpha2,
    # R=Rs,
    # norm_1=N1s,
)


u.save_analysis_tsv_special(
    run=simu,
    run_id=run_id,
    n=n,
    p=p,
    hp=hp_parameter,
    metric="fdr",
    alpha=alpha_list,
    fdr_selected=fdr_selected_dict,
    loss_train=loss_trains,
    loss_valid=loss_valids,
)
