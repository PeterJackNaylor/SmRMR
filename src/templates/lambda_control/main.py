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
from functools import partial
import yaml
import numpy as np
from jax import random
import jax.numpy as jnp
from sklearn.utils.validation import check_X_y
from scipy.sparse.linalg import eigsh

try:
    from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
except ImportError:
    from scipy.sparse.linalg import ArpackNoConvergence

from dclasso import DCLasso, pic_penalty
from dclasso.dc_lasso import loss

import utils as u
from lambda_control.lambda_control_utils import (
    build_iterator,
    fdr,
    perform_alpha_computations,
    perform_optimisation_with_parameters,
)

key = random.PRNGKey(42)
Cst = 1.5
max_epoch = 300
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

causal_feats = np.load("${CAUSAL_NPZ}")
causal_feats = list(causal_feats["featnames"][causal_feats["selected"]])

conservative = len(causal_feats) >= 5


# Hyper-parameters
############################
am = "${AM}"
kernel = "${KERNEL}" if am in am_kernels else ""

# hyper-parameters in the for loop
param_grid = {}
f = open("${PARAMS_FILE}")
param_grid = yaml.load(f, Loader=yaml.Loader)

penalty = param_grid["penalty"]
optimizer = param_grid["optimizer"]
lambdas = param_grid["lambda"]
n1 = param_grid["n1"]

iterable = build_iterator(penalty, optimizer, lambdas)

penalty_init, optimizer_init, lambda_init = next(iterable)

penalty_kwargs = {"name": penalty_init, "lamb": lambda_init}

# File parameters
alpha_list = np.arange(0.1, 1, 0.05)
fdr_alpha = []
selected_variables = []
loss_trains = []
loss_valids = []
Rs = []
N1s = []

penalties_parameter = []
optimizers__parameter = []
lambda_parameter = []
alpha_parameter = []


# default parameters

# Process

# One round to fit and get parameters

dl = DCLasso(alpha=alpha_list[0], measure_stat=am, kernel=kernel)

dl.fit(
    X,
    y,
    n1=n1,
    max_epoch=max_epoch,
    penalty_kwargs=penalty_kwargs,
    optimizer=optimizer[0],
    conservative=conservative,
)
loss_fn = partial(
    loss,
    Dxy=dl.Dxy,
    Dxx=dl.Dxx,
    penalty_func=pic_penalty({"name": "None"}),
)

loss_train__ = float(loss_fn(dl.beta_))
selected_ = list(np.array(dl.alpha_indices_))
fdr_ = fdr(causal_feats, selected_)

R__ = float(Cst / penalty_kwargs["lamb"] * pic_penalty(penalty_kwargs)(dl.beta_))
N1__ = np.abs(dl.beta_).sum()

# Compute validation loss, but only compute X_val, Xhat and Dxx/Dxy once
d = len(dl.screen_indices_)
X_val = X_val[:, dl.screen_indices_]
# Xhat = get_equi_features(X_val, key)
# X_val = jnp.concatenate([X_val, Xhat], axis=1)
y_val = jnp.array(y_val)

if "precompute" in dl.ms_kwargs.keys():
    del dl.ms_kwargs["precompute"]

Dxy_val = dl._compute_assoc(X_val, y_val, **dl.ms_kwargs)
Dxx_val = dl._compute_assoc(X_val, **dl.ms_kwargs)
dl.Dxy_val = Dxy_val
dl.Dxx_val = Dxx_val

# Compute validation loss fn
loss_fn = partial(
    loss, Dxy=Dxy_val, Dxx=Dxx_val, penalty_func=pic_penalty(penalty_kwargs)
)
loss_valid__ = float(loss_fn(dl.beta_[:d]))

fdr__, selected__ = perform_alpha_computations(
    alpha_list[1:], dl.wjs_, dl.screen_indices_, causal_feats, conservative
)

penalties_parameter += [penalty_init] * len(alpha_list)
optimizers__parameter += [optimizer_init] * len(alpha_list)
lambda_parameter += [lambda_init] * len(alpha_list)
alpha_parameter += list(alpha_list)
loss_trains += [loss_train__] * len(alpha_list)
loss_valids += [loss_valid__] * len(alpha_list)
Rs += [R__] * len(alpha_list)
N1s += [N1__] * len(alpha_list)
fdr_alpha += [fdr_] + fdr__
selected_variables += [selected_] + selected__


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
    penalties_parameter += [pen] * len(alpha_list)
    optimizers__parameter += [opt] * len(alpha_list)
    lambda_parameter += [lam] * len(alpha_list)
    alpha_parameter += list(alpha_list)
    loss_trains += [loss_train__] * len(alpha_list)
    loss_valids += [loss_valid__] * len(alpha_list)
    Rs += [R__] * len(alpha_list)
    N1s += [N1__] * len(alpha_list)
    fdr_alpha += fdr__
    selected_variables += selected__

# Some theoritical results that depends only on the data
try:
    alpha2 = float(eigsh(np.array(dl.Dxx), k=1, which="SA")[0].squeeze())
except ArpackNoConvergence:
    alpha2 = 0
# Save results
############################

simu, _ = "${TAG}".split("(")
u.save_analysis_tsv(
    run=simu,
    n=n,
    p=p,
    AM=am,
    kernel=kernel,
    penalty=penalties_parameter,
    optimizer=optimizers__parameter,
    metric="fdr",
    lamb=lambda_parameter,
    alpha=alpha_parameter,
    value=fdr_alpha,
    selected=selected_variables,
    loss_train=loss_trains,
    loss_valid=loss_valids,
    alpha2=alpha2,
    R=Rs,
    norm_1=N1s,
)
