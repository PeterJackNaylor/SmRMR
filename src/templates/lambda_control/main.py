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
import yaml
from functools import partial
from itertools import product
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
from dclasso.dc_lasso import alpha_threshold, loss, minimize_loss
from dclasso.utils import get_equi_features
import utils as u

key = random.PRNGKey(42)
Cst = 1.5


def fdr(causal_features, selected_features):

    if len(selected_features) == 0:
        print("No feature selection process happened")
        return -1
    else:
        n_selected = len(selected_features)
        intersection = list(set(selected_features) & set(causal_features))
        number_of_correct_positives = len(intersection)
        fdr = (n_selected - number_of_correct_positives) / n_selected
        return fdr


def perform_alpha_computations(alpha_list, wjs, screen_indices):
    run_fdr = []
    run_var_selected = []
    for alpha in alpha_list:

        selected_features, _, _ = alpha_threshold(alpha, wjs, screen_indices)
        selected_features = list(np.array(selected_features))
        run_fdr.append(fdr(causal_feats, selected_features))
        run_var_selected.append(selected_features)
    return run_fdr, run_var_selected


def perform_optimisation_with_parameters(dclasso_main, pen, opt, lam, alpha_list, d):
    penalty_kwargs = {"name": pen, "lamb": lam}
    loss_fn = partial(
        loss,
        Dxy=dclasso_main.Dxy,
        Dxx=dclasso_main.Dxx,
        penalty_func=pic_penalty(penalty_kwargs),
    )

    step_function, opt_state, beta = dclasso_main.setup_optimisation(
        loss_fn,
        opt,
        X_val.shape[1],
        key,
        "from_convex_solve",
        dclasso_main.Dxx,
        dclasso_main.Dxy,
        opt_kwargs,
    )
    beta, loss_train = minimize_loss(
        step_function,
        opt_state,
        beta,
        max_epoch,
        eps_stop,
        verbose=dclasso_main.verbose,
    )
    wjs = beta[:d] - beta[d:]
    fdr_l, selected_l = perform_alpha_computations(
        alpha_list, wjs, dclasso_main.screen_indices_
    )
    loss_fn = partial(
        loss,
        Dxy=dclasso_main.Dxy_val,
        Dxx=dclasso_main.Dxx_val,
        penalty_func=pic_penalty(penalty_kwargs),
    )
    loss_valid = float(loss_fn(beta))
    R = float(Cst / penalty_kwargs["lamb"] * pic_penalty(penalty_kwargs)(beta))
    N1 = np.abs(beta).sum()

    return fdr_l, selected_l, loss_train, loss_valid, R, N1


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

penalty_init = penalty[0]
optimizer_init = optimizer[0]
lambda_init = lambdas[0]

penalty_kwargs = {"name": penalty_init, "lamb": lambda_init}

# File parameters
alpha_list = np.arange(0.1, 1, 0.1)
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
opt_kwargs = {
    "init_value": 0.001,
    "transition_steps": 100,
    "decay_rate": 0.99,
}
max_epoch = 300
eps_stop = 1e-8

# Process

# One round to fit and get parameters

dl = DCLasso(alpha=alpha_list[0], measure_stat=am, kernel=kernel)

dl.fit(
    X,
    y,
    n1=0.5,
    max_epoch=max_epoch,
    penalty_kwargs=penalty_kwargs,
    optimizer=optimizer[0],
)
loss_train__ = float(dl.final_loss_)
selected_ = list(np.array(dl.alpha_indices_))
fdr_ = fdr(causal_feats, selected_)

R__ = float(Cst / penalty_kwargs["lamb"] * pic_penalty(penalty_kwargs)(dl.beta_))
N1__ = np.abs(dl.beta_).sum()

# Compute validation loss, but only compute X_val, Xhat and Dxx/Dxy once
d = len(dl.screen_indices_)
X_val = X_val[:, dl.screen_indices_]
Xhat = get_equi_features(X_val, key)
X_val = jnp.concatenate([X_val, Xhat], axis=1)
y_val = jnp.array(y_val)
Dxy_val = dl._compute_assoc(X_val, y_val, **dl.ms_kwargs)
Dxx_val = dl._compute_assoc(X_val, **dl.ms_kwargs)
dl.Dxy_val = Dxy_val
dl.Dxx_val = Dxx_val

# Compute validation loss fn
loss_fn = partial(
    loss, Dxy=Dxy_val, Dxx=Dxx_val, penalty_func=pic_penalty(penalty_kwargs)
)
loss_valid__ = float(loss_fn(dl.beta_))

fdr__, selected__ = perform_alpha_computations(
    alpha_list[1:], dl.wjs_, dl.screen_indices_
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
iterable = product(penalty, optimizer, lambdas)
next(iterable)
for pen, opt, lam in iterable:
    penalty_kwargs = {"name": pen, "lamb": lam}
    (
        fdr__,
        selected__,
        loss_train__,
        loss_valid__,
        R__,
        N1__,
    ) = perform_optimisation_with_parameters(dl, pen, opt, lam, alpha_list, d)
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
